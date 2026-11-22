"""SetFit: tune the encoder on sentence pairs (cosine MSE), then fit the shared LR.

Validation every eval_every updates: embed the training subset with the current
encoder, fit LR on training labels only, score validation macro-F1. The best
checkpoint (earliest on ties) is restored, the training subset is re-embedded and
the final LR is fitted. All of that counts toward training wall time.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .common import RunData, RunSpec, base_result, final_evaluation, timing_block
from .evaluate import compute_metrics, fit_logreg
from .pairs import build_setfit_pairs
from .utils import (TimeLedger, batches, linear_warmup_decay, log, make_grad_scaler,
                    peak_memory_mb, reset_peak_memory, set_seed, snapshot, wall_clock)


def run_setfit(spec: RunSpec, data: RunData, encoder_factory, device, use_amp: bool):
    set_seed(spec.seed)
    rng = np.random.default_rng(spec.seed)
    enc = encoder_factory()
    texts = data.train["text"].tolist()
    y_train = data.train["label_id"].to_numpy()
    val_texts = data.val["text"].tolist()
    y_val = data.val["label_id"].to_numpy()

    reset_peak_memory(device)
    T = TimeLedger(device)
    t0 = wall_clock(device)

    with T.section("pair_build"):
        a_idx, b_idx, z = build_setfit_pairs(y_train, rng, C.N_POS, C.N_NEG)

    opt = torch.optim.AdamW(enc.parameters(), lr=spec.lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, linear_warmup_decay(spec.max_updates, C.WARMUP_FRAC))
    scaler = make_grad_scaler(use_amp)
    stream = batches(len(z), spec.batch_size, rng)

    best_f1, best_step, best_state = -1.0, None, None
    history, loss_curve, window, since_eval = [], [], [], []
    for step in range(1, spec.max_updates + 1):
        idx = next(stream)
        with T.section("encoder_update"):
            enc.train()
            n = len(idx)
            emb = enc([texts[i] for i in a_idx[idx]] + [texts[i] for i in b_idx[idx]])  # one shared encoder
            cos = (emb[:n] * emb[n:]).sum(dim=-1)  # embeddings are unit-norm
            target = torch.as_tensor(z[idx], dtype=torch.float32, device=cos.device)
            loss = F.mse_loss(cos, target)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(enc.parameters(), C.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_value = float(loss.detach())
        window.append(loss_value)
        since_eval.append(loss_value)
        if step % 10 == 0 or step == spec.max_updates:
            loss_curve.append([step, float(np.mean(window))])
            window = []

        if step % spec.eval_every == 0:
            with T.section("checkpoint_train_embedding"):
                X_train = enc.encode(texts)
            with T.section("checkpoint_lr_fit"):
                clf, lr_info = fit_logreg(X_train, y_train)
            with T.section("checkpoint_validation"):
                m = compute_metrics(y_val, clf.predict(enc.encode(val_texts)), data.n_classes)
                if m["macro_f1"] > best_f1:  # strict: ties keep the earlier checkpoint
                    best_f1, best_step, best_state = m["macro_f1"], step, snapshot(enc)
            history.append({"step": step, "train_loss": float(np.mean(since_eval)),
                            "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"],
                            "lr": float(sched.get_last_lr()[0]), **lr_info})
            since_eval = []
            log(f"  {spec.run_id} step {step}: loss {history[-1]['train_loss']:.4f} "
                f"val macro-F1 {m['macro_f1']:.4f}")

    if best_state is None:
        raise ValueError("no validation checkpoint: max_updates must be >= eval_every")
    enc.load_state_dict(best_state)
    with T.section("final_train_embedding"):
        X_train = enc.encode(texts)
    with T.section("final_lr_fit"):
        clf, lr_info = fit_logreg(X_train, y_train)
    wall = wall_clock(device) - t0

    scores, preds = final_evaluation(enc, clf.predict, data, T)
    result = base_result(spec, data, device, use_amp, enc)
    result.update(scores)
    result.update(
        best_step=best_step,
        best_val_macro_f1=best_f1,
        n_pairs=int(len(z)),
        lr_checkpoint_fits_expected_max=spec.max_updates // spec.eval_every,
        lr_checkpoint_fits_observed=len(history),
        final_classifier=lr_info,
        timing=timing_block(T, wall, data.test is not None),
        peak_gpu_memory_mb=peak_memory_mb(device),
        history=history,
        loss_curve=loss_curve,
    )
    return result, preds
