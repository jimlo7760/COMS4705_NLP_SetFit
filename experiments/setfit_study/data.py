"""BANKING77: download, audit, validation split and nested few-shot samples.

Protocol (from the proposal):
  * strip outer whitespace only; exact-string comparison
  * drop redundant training rows, then drop training texts that also occur in the
    official test set (test *texts* only; test labels are never used for filtering)
  * keep all 3,080 official test rows
  * validation = floor(10%) of each class, seed 2026
  * for each seed, shuffle each class of the remaining pool once and take the first
    k in {8, 32} rows, so the 8-shot set is contained in the 32-shot set
"""
from __future__ import annotations

import datetime as dt
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .utils import log, read_json, sha256_file, write_json


# ----------------------------------------------------------------------------- download
def download_raw(raw_dir: Path, verify_hash: bool = True) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for split, fname in C.RAW_FILES.items():
        path = raw_dir / fname
        url = C.DATA_BASE_URL + fname
        if not path.exists():
            log(f"downloading {url}")
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)
        digest = sha256_file(path)
        info[split] = {"url": url, "path": str(path), "sha256": digest}
        if digest != C.RAW_SHA256[split]:
            msg = (f"{fname}: SHA-256 {digest} differs from the proposal audit "
                   f"({C.RAW_SHA256[split]}). The upstream file may have changed.")
            if verify_hash:
                raise RuntimeError(msg + " Re-run with --allow-hash-mismatch only after "
                                   "checking the audit counts, and mention it in the report.")
            log("WARNING: " + msg)
    return info


# ----------------------------------------------------------------------------- audit + splits
def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = {"text", "category"} - set(df.columns)
    if missing:
        raise ValueError(f"expected columns text, category; missing {missing}")
    out = df.rename(columns={"category": "label"}).reset_index(drop=True).copy()
    out["row_id"] = np.arange(len(out))
    out["text"] = out["text"].astype(str).str.strip()
    return out[["row_id", "text", "label"]]


def _check_expected(audit: dict, expected: dict) -> None:
    bad = {k: (audit.get(k), v) for k, v in expected.items() if audit.get(k) != v}
    if bad:
        lines = "\n".join(f"  {k}: got {got}, expected {want}" for k, (got, want) in bad.items())
        raise AssertionError("data audit does not match the proposal:\n" + lines +
                             "\nDo not edit the expected numbers to hide this; find out why.")


def build_splits(train_raw: pd.DataFrame, test_raw: pd.DataFrame, expected: dict | None = None) -> dict:
    train = _prepare_frame(train_raw)
    test = _prepare_frame(test_raw)

    audit = {"original_train": len(train), "original_test": len(test)}
    audit["label_conflicts"] = int((train.groupby("text")["label"].nunique() > 1).sum())
    audit["redundant_train_rows"] = int(train.duplicated("text").sum())
    overlap = sorted(set(train["text"]) & set(test["text"]))     # texts only
    audit["cross_split_train_texts"] = len(overlap)
    audit["test_duplicate_rows"] = int(test.duplicated("text").sum())

    clean = train.drop_duplicates("text", keep="first")
    clean = clean[~clean["text"].isin(overlap)].copy()
    audit["clean_train"] = len(clean)

    labels = sorted(clean["label"].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}
    audit["n_classes"] = len(labels)
    unknown = sorted(set(test["label"]) - set(label2id))
    if unknown:
        raise ValueError(f"test labels missing from training: {unknown[:5]}")
    clean["label_id"] = clean["label"].map(label2id).astype(int)
    test["label_id"] = test["label"].map(label2id).astype(int)

    rng = np.random.default_rng(C.VAL_SEED)
    val_idx = []
    for _, part in clean.groupby("label", sort=True):
        ids = rng.permutation(part.index.to_numpy())
        val_idx.extend(ids[: len(part) // 10].tolist())
    val = clean.loc[val_idx].copy()
    pool = clean.drop(index=val_idx).copy()
    audit["validation"] = len(val)
    audit["pool"] = len(pool)
    audit["min_pool_per_class"] = int(pool.groupby("label").size().min())

    if expected:
        _check_expected(audit, expected)
    return {"clean": clean, "val": val, "pool": pool, "test": test,
            "labels": labels, "overlap_texts": overlap, "audit": audit}


def make_nested_samples(pool: pd.DataFrame, seed: int, shots=C.SHOTS) -> dict:
    rng = np.random.default_rng(seed)
    chosen = {k: [] for k in shots}
    for label, part in pool.groupby("label", sort=True):
        ids = rng.permutation(part.index.to_numpy())
        for k in shots:
            if len(ids) < k:
                raise ValueError(f"class {label} has {len(ids)} pool rows < {k}")
            chosen[k].extend(ids[:k].tolist())
    return {k: pool.loc[ids].copy() for k, ids in chosen.items()}


# ----------------------------------------------------------------------------- files
COLUMNS = ["row_id", "text", "label", "label_id"]


def sample_seeds() -> tuple:
    return (C.TUNING_SEED,) + tuple(C.FINAL_SEEDS)


def train_file(data_dir: Path, k: int, seed: int) -> Path:
    return data_dir / f"train_k{k}_seed{seed}.csv"


def _split_files(data_dir: Path) -> list:
    files = [data_dir / "val.csv", data_dir / "test.csv", data_dir / "label_map.json"]
    files += [train_file(data_dir, k, s) for s in sample_seeds() for k in C.SHOTS]
    return files


def prepare_data(root, expected: dict | None = C.EXPECTED_COUNTS, verify_hash: bool = True,
                 force: bool = False) -> dict:
    """Create (or verify) every split file under <root>/data. Idempotent."""
    data_dir = Path(root) / "data"
    manifest_path = data_dir / "split_manifest.json"
    if manifest_path.exists() and not force:
        manifest = read_json(manifest_path)
        changed = [n for n, digest in manifest["files"].items()
                   if not (data_dir / n).exists() or sha256_file(data_dir / n) != digest]
        if changed:
            raise RuntimeError(f"split files differ from {manifest_path.name}: {changed}. "
                               "Restore them, or re-create everything with --force-data "
                               "(only before any final run).")
        log(f"data: {len(manifest['files'])} split files verified against the manifest")
        return manifest

    raw = download_raw(data_dir / "raw", verify_hash=verify_hash)
    train_raw = pd.read_csv(data_dir / "raw" / C.RAW_FILES["train"], keep_default_na=False)
    test_raw = pd.read_csv(data_dir / "raw" / C.RAW_FILES["test"], keep_default_na=False)
    res = build_splits(train_raw, test_raw, expected)

    res["val"][COLUMNS].to_csv(data_dir / "val.csv", index=False)
    res["test"][COLUMNS].to_csv(data_dir / "test.csv", index=False)
    write_json(data_dir / "label_map.json", res["labels"])
    for seed in sample_seeds():
        samples = make_nested_samples(res["pool"], seed)
        assert set(samples[min(C.SHOTS)]["row_id"]) <= set(samples[max(C.SHOTS)]["row_id"])
        for k, df in samples.items():
            df[COLUMNS].to_csv(train_file(data_dir, k, seed), index=False)

    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "protocol": "strip outer whitespace; drop redundant train rows; drop train texts "
                    "that occur in test (texts only); val = floor(10%) per class (seed "
                    f"{C.VAL_SEED}); nested per-class samples for seeds {list(sample_seeds())}",
        "raw": raw,
        "audit": res["audit"],
        "overlap_texts_removed_from_train": res["overlap_texts"],
        "files": {p.name: sha256_file(p) for p in _split_files(data_dir)},
    }
    write_json(manifest_path, manifest)
    log("data audit: " + ", ".join(f"{k}={v}" for k, v in res["audit"].items()))
    return manifest


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, dtype={"text": str, "label": str})
    return df[COLUMNS]


def load_split(root, name: str) -> pd.DataFrame:
    return _read_csv(Path(root) / "data" / f"{name}.csv")


def load_train(root, k: int, seed: int) -> pd.DataFrame:
    return _read_csv(train_file(Path(root) / "data", k, seed))


def load_labels(root) -> list:
    return read_json(Path(root) / "data" / "label_map.json")
