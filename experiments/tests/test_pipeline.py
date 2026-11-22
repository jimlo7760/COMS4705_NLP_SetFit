"""Offline tests: synthetic BANKING77-shaped data and a tiny stand-in encoder.

Covers the data protocol, pair sampler, schedule, all three training loops and every
orchestrator stage. The Hugging Face encoder and the CUDA/FP16 path are NOT exercised
here (no downloads); the Colab smoke stage covers those.

Run:  cd experiments && python -m pytest -q tests
"""
from __future__ import annotations

import os
import sys
import zlib
from pathlib import Path

os.environ.setdefault("SETFIT_DEVICE", "cpu")  # the stand-in encoder lives on CPU, even on a GPU machine

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from setfit_study import config as C  # noqa: E402
from setfit_study import report, run_all  # noqa: E402
from setfit_study.data import build_splits, make_nested_samples  # noqa: E402
from setfit_study.pairs import build_setfit_pairs  # noqa: E402
from setfit_study.utils import linear_warmup_decay, read_json  # noqa: E402

N_CLASSES = 77


class DummyEncoder(nn.Module):
    """Hashed bag-of-words -> 32-d unit vectors; same interface as SentenceEncoder."""
    dim = 32

    def __init__(self, buckets: int = 2048):
        super().__init__()
        self.buckets = buckets
        self.emb = nn.EmbeddingBag(buckets, self.dim, mode="mean")
        self.proj = nn.Linear(self.dim, self.dim)
        self.device = torch.device("cpu")
        self.commit = None

    def forward(self, texts):
        ids, offsets = [], []
        for t in texts:
            offsets.append(len(ids))
            ids.extend(zlib.crc32(w.encode()) % self.buckets for w in (t.split() or ["<e>"]))
        h = self.proj(torch.tanh(self.emb(torch.tensor(ids), torch.tensor(offsets))))
        return F.normalize(h, dim=-1)

    @torch.no_grad()
    def encode(self, texts, batch_size: int = 256):
        was = self.training
        self.eval()
        out = self.forward(list(texts)).numpy().astype(np.float32)
        self.train(was)
        return out

    def truncation_rate(self, texts):
        return 0.0


def synthetic_raw(seed: int = 0, per_class: int = 130, test_per_class: int = 40):
    rng = np.random.default_rng(seed)
    counter = iter(range(10**7))

    def text(c):
        words = [f"c{c}w{j}" for j in rng.choice(8, size=3, replace=False)]
        words += [f"n{j}" for j in rng.choice(60, size=3)]
        rng.shuffle(words)
        return " ".join(words + [f"id{next(counter)}"])

    labels = [f"intent_{c:02d}" for c in range(N_CLASSES)]
    train = pd.DataFrame([(text(c), labels[c]) for c in range(N_CLASSES) for _ in range(per_class)],
                         columns=["text", "category"])
    test = pd.DataFrame([(text(c), labels[c]) for c in range(N_CLASSES) for _ in range(test_per_class)],
                        columns=["text", "category"])
    # 4 redundant training rows (one differs only by outer whitespace)
    dups = train.iloc[[3, 500, 1000, 2000]].copy()
    dups.iloc[0, 0] = "  " + dups.iloc[0, 0] + " \n"
    train = pd.concat([train, dups], ignore_index=True)
    # 6 training texts that also appear in test
    for j, i in enumerate([10, 700, 1400, 3000, 5000, 9000]):
        test.iloc[j * 40, 0] = train.iloc[i, 0]
        test.iloc[j * 40, 1] = train.iloc[i, 1]
    return train, test


# ----------------------------------------------------------------------------- unit tests
def test_build_splits_protocol():
    train_raw, test_raw = synthetic_raw()
    res = build_splits(train_raw, test_raw, expected=None)
    a = res["audit"]
    assert a["redundant_train_rows"] == 4 and a["cross_split_train_texts"] == 6
    assert a["label_conflicts"] == 0 and a["n_classes"] == N_CLASSES
    assert a["clean_train"] == len(train_raw) - 4 - 6
    assert len(res["test"]) == len(test_raw)                      # test untouched
    assert not set(res["clean"]["text"]) & set(res["test"]["text"])
    per_class = res["clean"].groupby("label").size()
    assert a["validation"] == int((per_class // 10).sum())
    assert set(res["val"].index).isdisjoint(res["pool"].index)
    assert len(res["val"]) + len(res["pool"]) == len(res["clean"])

    s1, s2 = make_nested_samples(res["pool"], 11), make_nested_samples(res["pool"], 11)
    assert list(s1[8]["row_id"]) == list(s2[8]["row_id"])        # deterministic
    assert set(s1[8]["row_id"]) <= set(s1[32]["row_id"])           # nested
    assert (s1[8].groupby("label").size() == 8).all() and (s1[32].groupby("label").size() == 32).all()
    assert list(make_nested_samples(res["pool"], 22)[8]["row_id"]) != list(s1[8]["row_id"])


def test_expected_counts_are_enforced():
    train_raw, test_raw = synthetic_raw()
    with pytest.raises(AssertionError, match="clean_train"):
        build_splits(train_raw, test_raw, expected={"clean_train": 9993})


def test_pairs():
    labels = np.repeat(np.arange(N_CLASSES), 8)
    a, b, z = build_setfit_pairs(labels, np.random.default_rng(0))
    assert len(z) == 8 * len(labels) == 4928
    assert (a != b).all()
    assert ((labels[a] == labels[b]) == (z == 1)).all()
    assert z.sum() == 4 * len(labels)
    for i in range(len(labels)):                                 # 8 distinct partners per anchor
        assert len(set(b[a == i])) == 8
    a2, b2, z2 = build_setfit_pairs(labels, np.random.default_rng(0))
    assert (a == a2).all() and (b == b2).all()


def test_schedule():
    f = linear_warmup_decay(300, 0.1)
    assert f(0) == pytest.approx(1 / 30) and f(29) == pytest.approx(1.0)
    assert f(30) == pytest.approx(1.0) and 0 < f(299) < 0.01


def test_loops_learn():
    """With a learning rate suited to the stand-in encoder, both loops must beat Frozen+LR."""
    from setfit_study.common import RunData, RunSpec
    from setfit_study.train_ce import run_ce
    from setfit_study.train_frozen import run_frozen
    from setfit_study.train_setfit import run_setfit

    train_raw, test_raw = synthetic_raw()
    res = build_splits(train_raw, test_raw, expected=None)
    data = RunData(make_nested_samples(res["pool"], 11)[8], res["val"], None, res["labels"])
    dev = torch.device("cpu")
    frozen, _ = run_frozen(RunSpec(C.FROZEN, 8, 11, "tuning"), data, DummyEncoder, dev, False)
    setfit, _ = run_setfit(RunSpec(C.SETFIT, 8, 11, "tuning", lr=3e-2, max_updates=150, eval_every=50),
                           data, DummyEncoder, dev, False)
    ce, _ = run_ce(RunSpec(C.CE, 8, 11, "tuning", lr=3e-2, head_lr=5e-2, max_updates=150, eval_every=50),
                   data, DummyEncoder, dev, False)
    assert setfit["history"][-1]["train_loss"] < setfit["history"][0]["train_loss"]
    assert ce["history"][-1]["train_loss"] < ce["history"][0]["train_loss"]
    assert setfit["best_val_macro_f1"] > frozen["val_macro_f1"] + 0.2
    assert ce["best_val_macro_f1"] > frozen["val_macro_f1"] + 0.2
    # the locked model is the best checkpoint: re-scoring it reproduces the selection score
    assert setfit["val_macro_f1"] == pytest.approx(setfit["best_val_macro_f1"], abs=1e-9)
    assert ce["val_macro_f1"] == pytest.approx(ce["best_val_macro_f1"], abs=1e-9)


# ----------------------------------------------------------------------------- end to end
@pytest.fixture()
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MAX_UPDATES", {8: 20, 32: 30})
    monkeypatch.setattr(C, "EVAL_EVERY", {8: 10, 32: 15})
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    train_raw, test_raw = synthetic_raw()
    train_raw.to_csv(raw / "train.csv", index=False)
    test_raw.to_csv(raw / "test.csv", index=False)
    return tmp_path


def run(root, *args):
    torch.manual_seed(0)
    return run_all.main(["--root", str(root), *args], encoder_factory=DummyEncoder,
                        expected=None, verify_hash=False)


def test_all_stages(project):
    root = project
    res = root / "results"
    assert run(root, "--stage", "smoke") == 0
    assert len(list((res / "smoke" / "runs").glob("*.json"))) == 3
    assert not (res / "runs").exists()                               # smoke kept apart

    assert run(root, "--stage", "frozen") == 0
    frozen = report.load_runs(res)
    assert len(frozen) == 8
    assert all((r["test_macro_f1"] is None) == (r["phase"] == "tuning") for r in frozen)
    assert not list((res / "preds").glob("*seed0*_test.csv"))         # tuning never touches test
    pairs = read_json(res / "confusable_pairs.json")
    assert len(pairs["pairs"]) <= 5

    assert run(root, "--stage", "tune") == 0
    tuning = [r for r in report.load_runs(res) if r["phase"] == "tuning" and r["method"] != C.FROZEN]
    assert len(tuning) == 8
    for r in tuning:
        assert r["test_macro_f1"] is None
        if r["method"] == C.SETFIT:
            assert r["lr_checkpoint_fits_observed"] == r["lr_checkpoint_fits_expected_max"] == 2
            assert r["n_pairs"] == 8 * r["n_train"]
        assert len(r["history"]) == 2 and r["best_step"] in (10, 15, 20, 30)

    with pytest.raises(SystemExit):                                   # final before select
        run(root, "--stage", "final")
    assert run(root, "--stage", "select") == 0
    sel = read_json(res / "selected_hparams.json")
    assert set(sel[C.SETFIT]) == {"8", "32"}
    assert run(root, "--stage", "select") == 0                        # locked: unchanged
    assert read_json(res / "selected_hparams.json")[C.CE] == sel[C.CE]

    assert run(root, "--stage", "final", "--shots", "8") == 0
    assert run(root, "--stage", "final") == 0
    finals = [r for r in report.load_runs(res) if r["phase"] == "final"]
    assert len(finals) == 18 and all(r["sample_and_optimizer_seed"] in C.FINAL_SEEDS for r in finals)
    assert all(r["test_macro_f1"] is not None for r in finals)
    setfit_final = [r for r in finals if r["method"] == C.SETFIT]
    assert all(r["learning_rate"] == sel[C.SETFIT][str(r["shots_per_intent"])]["lr"] for r in setfit_final)

    mtimes = {p: p.stat().st_mtime for p in (res / "runs").glob("*.json")}
    assert run(root, "--stage", "final") == 0                         # resume: nothing re-run
    assert mtimes == {p: p.stat().st_mtime for p in (res / "runs").glob("*.json")}

    out = report.collect(root)
    assert len(out["summary"]) == 6 and len(out["paired"]) == 18
    log = pd.read_csv(res / "results_log.csv")
    assert list(log.columns[:len(report.TEMPLATE_COLUMNS)]) == report.TEMPLATE_COLUMNS
    assert (res / "figures" / "tuning_curves.png").exists()
    assert "Frozen+LR" in (res / "table_main.tex").read_text()
    assert len(out["pairs"]) == 18 * len(pairs["pairs"])


def test_final_refuses_second_setting(project):
    root = project
    from setfit_study.utils import write_json
    assert run(root, "--stage", "tune", "--shots", "8", "--methods", "setfit") == 0
    with pytest.raises(SystemExit):                    # CE not tuned yet
        run(root, "--stage", "select", "--shots", "8")
    assert run(root, "--stage", "select", "--shots", "8", "--methods", "setfit") == 0
    sel_path = root / "results" / "selected_hparams.json"
    locked = read_json(sel_path)[C.SETFIT]["8"]["lr"]
    assert run(root, "--stage", "final", "--shots", "8", "--methods", "setfit", "--seeds", "11") == 0
    # tampering with the lock must not produce a second test score for the same setting
    other = [lr for lr in C.LEARNING_RATES if lr != locked][0]
    write_json(sel_path, {C.SETFIT: {"8": {"lr": other}}})
    assert run(root, "--stage", "final", "--shots", "8", "--methods", "setfit", "--seeds", "11") == 1


def test_ce_head_lr_variant(project):
    root = project
    assert run(root, "--stage", "tune", "--shots", "8", "--methods", "ce", "--ce-head-lr", "1e-3") == 0
    runs = [r for r in report.load_runs(root / "results") if r["method"] == C.CE]
    assert {r["head_lr"] for r in runs} == {1e-3} and all("hlr0.001" in r["run_id"] for r in runs)
    with pytest.raises(SystemExit):   # no default-head runs to select from
        run(root, "--stage", "select", "--shots", "8")
