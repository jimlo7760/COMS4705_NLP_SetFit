"""Metrics, the shared logistic-regression classifier, and prediction tables."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

from . import config as C


def compute_metrics(y_true, y_pred, n_classes: int) -> dict:
    """Macro-F1 over all classes (undefined F1 = 0), accuracy, per-class recall."""
    labels = np.arange(n_classes)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_recall": [round(float(x), 4) for x in recall],
    }


def fit_logreg(X, y):
    """The single LR configuration shared by Frozen+LR and SetFit (lbfgs, C=1, intercept)."""
    clf = LogisticRegression(**C.LR_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence is recorded explicitly below
        clf.fit(X, y)
    n_iter = int(np.max(clf.n_iter_))
    return clf, {"lr_n_iter": n_iter, "lr_converged": bool(n_iter < C.LR_PARAMS["max_iter"])}


def predictions_frame(df: pd.DataFrame, y_pred, labels: list) -> pd.DataFrame:
    y_pred = np.asarray(y_pred, dtype=int)
    return pd.DataFrame({
        "row_id": df["row_id"].to_numpy(),
        "true_id": df["label_id"].to_numpy(),
        "pred_id": y_pred,
        "true_label": df["label"].to_numpy(),
        "pred_label": [labels[i] for i in y_pred],
    })
