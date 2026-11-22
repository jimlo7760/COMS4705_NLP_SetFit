"""Frozen+LR: embed with the untouched encoder, fit the shared logistic regression."""
from __future__ import annotations

from .common import RunData, RunSpec, base_result, final_evaluation, timing_block
from .evaluate import fit_logreg
from .utils import TimeLedger, peak_memory_mb, reset_peak_memory, set_seed, wall_clock


def run_frozen(spec: RunSpec, data: RunData, encoder_factory, device, use_amp: bool):
    set_seed(spec.seed)
    enc = encoder_factory()
    texts = data.train["text"].tolist()
    y = data.train["label_id"].to_numpy()

    reset_peak_memory(device)
    T = TimeLedger(device)
    t0 = wall_clock(device)
    with T.section("final_train_embedding"):
        X = enc.encode(texts)
    with T.section("final_lr_fit"):
        clf, lr_info = fit_logreg(X, y)
    wall = wall_clock(device) - t0

    scores, preds = final_evaluation(enc, clf.predict, data, T)
    result = base_result(spec, data, device, use_amp, enc)
    result.update(scores)
    result.update(
        best_step=None,
        best_val_macro_f1=scores["val_macro_f1"],
        lr_checkpoint_fits_expected_max=0,
        lr_checkpoint_fits_observed=0,
        final_classifier=lr_info,
        timing=timing_block(T, wall, data.test is not None),
        peak_gpu_memory_mb=peak_memory_mb(device),
        history=[],
        loss_curve=[],
    )
    return result, preds
