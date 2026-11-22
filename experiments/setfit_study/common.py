"""Run specification, run data, and the result fields every method shares."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import config as C
from .evaluate import compute_metrics, predictions_frame
from .utils import TimeLedger, env_info


@dataclass
class RunSpec:
    method: str                      # C.FROZEN | C.CE | C.SETFIT
    k: int                           # training examples per intent
    seed: int                        # controls the sample, pairs, batch order and init
    phase: str                       # "tuning" (validation only) | "final" (test once) | "smoke"
    lr: Optional[float] = None       # encoder learning rate (CE, SetFit)
    head_lr: Optional[float] = None  # CE only; None = same as lr (proposal default)
    max_updates: int = 0
    eval_every: int = 0
    batch_size: int = C.BATCH_SIZE

    @property
    def run_id(self) -> str:
        parts = [C.METHOD_SLUG[self.method], f"k{self.k}", f"seed{self.seed}"]
        if self.lr is not None:
            parts.append(f"lr{self.lr:g}")
        if self.head_lr is not None:
            parts.append(f"hlr{self.head_lr:g}")
        parts.append(self.phase)
        return "_".join(parts)


@dataclass
class RunData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: Optional[pd.DataFrame]     # None for tuning runs: the test set is not touched
    labels: list

    @property
    def n_classes(self) -> int:
        return len(self.labels)


def base_result(spec: RunSpec, data: RunData, device, use_amp: bool, enc) -> dict:
    trunc = getattr(enc, "truncation_rate", None)
    splits = {"train": data.train, "val": data.val, "test": data.test}
    return {
        "run_id": spec.run_id,
        "phase": spec.phase,
        "method": spec.method,
        "shots_per_intent": spec.k,
        "sample_and_optimizer_seed": spec.seed,
        "learning_rate": spec.lr,
        "head_lr": spec.head_lr,
        "max_updates": spec.max_updates,
        "eval_every": spec.eval_every,
        "batch_size": spec.batch_size,
        "n_train": len(data.train),
        "n_val": len(data.val),
        "n_test": None if data.test is None else len(data.test),
        "model_id": C.MODEL_ID,
        "model_commit": getattr(enc, "commit", None),
        "amp_fp16": bool(use_amp),
        "truncation_rate": {name: (trunc(df["text"].tolist()) if trunc and df is not None else None)
                            for name, df in splits.items()},
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "env": env_info(device),
    }


def timing_block(T: TimeLedger, wall_sec: float, has_test: bool) -> dict:
    t = {
        "training_wall_sec_total": wall_sec,  # from pair building / first update to the final classifier
        "pair_build_sec": T.get("pair_build"),
        "encoder_update_sec": T.get("encoder_update"),
        "checkpoint_train_embedding_sec": T.get("checkpoint_train_embedding"),
        "checkpoint_lr_fit_sec": T.get("checkpoint_lr_fit"),
        "checkpoint_validation_sec": T.get("checkpoint_validation"),
        "final_train_embedding_sec": T.get("final_train_embedding"),
        "final_lr_fit_sec": T.get("final_lr_fit"),
        "test_inference_sec_separate": T.get("test_inference") if has_test else None,
    }
    return {k: (None if v is None else round(float(v), 4)) for k, v in t.items()}


def final_evaluation(enc, predict, data: RunData, T: TimeLedger):
    """Score the locked model on validation and (final runs only) once on test."""
    out = {"test_macro_f1": None, "test_accuracy": None, "test_per_class_recall": None}
    preds = {}
    y_val = data.val["label_id"].to_numpy()
    p_val = np.asarray(predict(enc.encode(data.val["text"].tolist())))
    m = compute_metrics(y_val, p_val, data.n_classes)
    out.update(val_macro_f1=m["macro_f1"], val_accuracy=m["accuracy"])
    preds["val"] = predictions_frame(data.val, p_val, data.labels)
    if data.test is not None:
        with T.section("test_inference"):
            p_test = np.asarray(predict(enc.encode(data.test["text"].tolist())))
        m = compute_metrics(data.test["label_id"].to_numpy(), p_test, data.n_classes)
        out.update(test_macro_f1=m["macro_f1"], test_accuracy=m["accuracy"],
                   test_per_class_recall=m["per_class_recall"])
        preds["test"] = predictions_frame(data.test, p_test, data.labels)
    return out, preds
