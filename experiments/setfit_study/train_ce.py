"""CE tuning: jointly train the encoder and a 77-way linear head with cross-entropy.

The head sits on the same unit-normalised sentence vector the other methods use.
Validation every eval_every updates scores the model's own head (no LR fits).

Watch the first tuning run: with a randomly initialised head on unit-norm inputs and
encoder-sized learning rates (1e-5 / 3e-5), the head can learn very slowly. If the
training loss stays near ln(77) ~ 4.34, use --ce-head-lr (e.g. 1e-3) for the head and
report that as a change since the proposal.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import config as C
from .common import RunData, RunSpec, base_result, final_evaluation, timing_block
from .evaluate import compute_metrics
from .utils import (TimeLedger, batches, linear_warmup_decay, log, make_grad_scaler,
                    peak_memory_mb, reset_peak_memory, set_seed, snapshot, wall_clock)


def _head_predict(head: nn.Module, device):
    def predict(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = head(torch.as_tensor(X, dtype=torch.float32, device=device))
        return logits.argmax(dim=-1).cpu().numpy()
    return predict


def run_ce(spec: RunSpec, data: RunData, encoder_factory, device, use_amp: bool):
    set_seed(spec.seed)
    rng = np.random.default_rng(spec.seed)
    enc = encoder_factory()
    head = nn.Linear(enc.dim, data.n_classes).to(device)   # initialised under the run seed
    model = nn.ModuleDict({"encoder": enc, "head": head})
    texts = data.train["text"].tolist()
    y_train = torch.as_tensor(data.train["label_id"].to_numpy(), dtype=torch.long, device=device)
    val_texts = data.val["text"].tolist()
    y_val = data.val["label_id"].to_numpy()
    predict = _head_predict(head, device)

    reset_peak_memory(device)
    T = TimeLedger(device)
    t0 = wall_clock(device)

    head_lr = spec.head_lr if spec.head_lr is not None else spec.lr
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": spec.lr},
                             {"params": head.parameters(), "lr": head_lr}],
                            lr=spec.lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, linear_warmup_decay(spec.max_updates, C.WARMUP_FRAC))
    scaler = make_grad_scaler(use_amp)
    stream = batches(len(texts), spec.batch_size, rng)

    best_f1, best_step, best_state = -1.0, None, None
    history, loss_curve, window, since_eval = [], [], [], []
    for step in range(1, spec.max_updates + 1):
        idx = next(stream)
        with T.section("encoder_update"):
            model.train()
            logits = head(enc([texts[i] for i in idx]))     # head in FP32
            loss = F.cross_entropy(logits, y_train[torch.as_tensor(idx, device=device)])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
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
            with T.section("checkpoint_validation"):
                m = compute_metrics(y_val, predict(enc.encode(val_texts)), data.n_classes)
                if m["macro_f1"] > best_f1:  # strict: ties keep the earlier checkpoint
                    best_f1, best_step, best_state = m["macro_f1"], step, snapshot(model)
            history.append({"step": step, "train_loss": float(np.mean(since_eval)),
                            "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"],
                            "lr": float(sched.get_last_lr()[0])})
            since_eval = []
            log(f"  {spec.run_id} step {step}: loss {history[-1]['train_loss']:.4f} "
                f"val macro-F1 {m['macro_f1']:.4f}")

    if best_state is None:
        raise ValueError("no validation checkpoint: max_updates must be >= eval_every")
    model.load_state_dict(best_state)
    wall = wall_clock(device) - t0

    scores, preds = final_evaluation(enc, predict, data, T)
    result = base_result(spec, data, device, use_amp, enc)
    result.update(scores)
    result.update(
        best_step=best_step,
        best_val_macro_f1=best_f1,
        effective_head_lr=head_lr,
        lr_checkpoint_fits_expected_max=0,
        lr_checkpoint_fits_observed=0,
        timing=timing_block(T, wall, data.test is not None),
        peak_gpu_memory_mb=peak_memory_mb(device),
        history=history,
        loss_curve=loss_curve,
    )
    return result, preds
