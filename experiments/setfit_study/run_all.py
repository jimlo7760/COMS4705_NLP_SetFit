"""Orchestrator:  python -m setfit_study.run_all --root <DIR> --stage <STAGE> [options]

Stages, in order:
  smoke    10-update CE/SetFit plus Frozen+LR (8-shot, seed 0) to check the whole pipeline.
           Writes to results/smoke; a validation subset stands in for the test set.
  data     download, audit and split (every run stage also does this, idempotently)
  frozen   Frozen+LR for both sizes: seed 0 (validation only) and seeds 11/22/33 (test once).
           Also locks the five confusable intent pairs from seed-0 8-shot validation.
  tune     CE and SetFit, seed 0, both learning rates (validation only)
  select   lock one learning rate per method and size -> results/selected_hparams.json
  final    CE and SetFit (and any missing Frozen+LR) on seeds 11/22/33, each scored once on test
  collect  results_log.csv, summary tables, LaTeX table, training-curve figure

A run is finished when results/runs/<run_id>.json exists. Finished runs are skipped,
so after a Colab disconnect just run the same command again.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch

from . import config as C
from . import report
from .common import RunData, RunSpec
from .data import load_labels, load_split, load_train, prepare_data
from .models import SentenceEncoder
from .train_ce import run_ce
from .train_frozen import run_frozen
from .train_setfit import run_setfit
from .utils import env_info, get_device, log, read_json, write_json

RUNNERS = {C.FROZEN: run_frozen, C.CE: run_ce, C.SETFIT: run_setfit}
STAGES = ("smoke", "data", "frozen", "tune", "select", "final", "collect")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="project folder that holds data/ and results/")
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--shots", type=int, nargs="+", choices=list(C.SHOTS), help="default: 8 and 32")
    p.add_argument("--seeds", type=int, nargs="+", help="restrict seeds (e.g. to split work)")
    p.add_argument("--methods", nargs="+", choices=list(C.SLUG_METHOD), help="frozen ce setfit")
    p.add_argument("--ce-head-lr", type=float, default=None,
                   help="CE only: separate learning rate for the new 77-way head "
                        "(default: same as the encoder, as in the proposal)")
    p.add_argument("--force", action="store_true",
                   help="re-run finished runs, or overwrite a locked selection")
    p.add_argument("--force-data", action="store_true", help="rebuild data/ (never after final runs)")
    p.add_argument("--allow-hash-mismatch", action="store_true")
    p.add_argument("--no-amp", action="store_true", help="disable FP16 autocast on GPU")
    p.add_argument("--fail-fast", action="store_true", help="stop at the first failed run")
    p.add_argument("--dry-run", action="store_true", help="list the planned runs and exit")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------- planning
def _methods(args, allowed):
    chosen = [C.SLUG_METHOD[s] for s in args.methods] if args.methods else list(allowed)
    return [m for m in C.METHODS if m in chosen and m in allowed]


def selected_path(root) -> Path:
    return Path(root) / "results" / "selected_hparams.json"


def plan(stage: str, args, root) -> list:
    shots = sorted(args.shots or C.SHOTS)
    if stage == "smoke":
        return [RunSpec(m, min(C.SHOTS), C.TUNING_SEED, "smoke",
                        lr=None if m == C.FROZEN else max(C.LEARNING_RATES),
                        head_lr=args.ce_head_lr if m == C.CE else None,
                        max_updates=10, eval_every=5) for m in C.METHODS]
    if stage == "frozen":
        seeds = args.seeds or [C.TUNING_SEED, *C.FINAL_SEEDS]
        return [RunSpec(C.FROZEN, k, s, "tuning" if s == C.TUNING_SEED else "final")
                for k in shots for s in seeds]
    if stage == "tune":
        if args.seeds and args.seeds != [C.TUNING_SEED]:
            raise SystemExit(f"tuning uses seed {C.TUNING_SEED} only")
        return [RunSpec(m, k, C.TUNING_SEED, "tuning", lr=lr,
                        head_lr=args.ce_head_lr if m == C.CE else None,
                        max_updates=C.MAX_UPDATES[k], eval_every=C.EVAL_EVERY[k])
                for k in shots for m in _methods(args, (C.CE, C.SETFIT)) for lr in C.LEARNING_RATES]
    if stage == "final":
        seeds = args.seeds or list(C.FINAL_SEEDS)
        if C.TUNING_SEED in seeds:
            raise SystemExit(f"seed {C.TUNING_SEED} is the tuning seed; it is never scored on test")
        methods = _methods(args, C.METHODS)
        selected = read_json(selected_path(root)) if selected_path(root).exists() else {}
        specs = []
        for k in shots:
            for s in seeds:
                for m in methods:
                    if m == C.FROZEN:
                        specs.append(RunSpec(C.FROZEN, k, s, "final"))
                        continue
                    entry = selected.get(m, {}).get(str(k))
                    if entry is None:
                        raise SystemExit(f"no locked learning rate for {m}, {k}-shot: "
                                         "run --stage tune, then --stage select")
                    specs.append(RunSpec(m, k, s, "final", lr=entry["lr"],
                                         head_lr=entry.get("head_lr") if m == C.CE else None,
                                         max_updates=C.MAX_UPDATES[k], eval_every=C.EVAL_EVERY[k]))
        return specs
    raise ValueError(stage)


# ----------------------------------------------------------------------------- selection
def select_hparams(root, shots, ce_head_lr=None, force=False, methods=(C.CE, C.SETFIT)) -> dict:
    """Highest best-checkpoint validation macro-F1 on the seed-0 tuning runs; ties -> smaller lr.
    Each (method, size) entry is locked once written; --force overwrites it."""
    path = selected_path(root)
    selected = read_json(path) if path.exists() else {
        "rule": "highest best_val_macro_f1 over the seed-0 tuning runs; ties -> smaller learning rate"}
    runs = report.load_runs(report.results_dir(root), phases=("tuning",))
    for m in methods:
        for k in sorted(shots):
            if str(k) in selected.get(m, {}) and not force:
                log(f"{m}, {k}-shot: already locked at lr={selected[m][str(k)]['lr']:g}")
                continue
            cands = {r["learning_rate"]: r["best_val_macro_f1"] for r in runs
                     if r["method"] == m and r["shots_per_intent"] == k
                     and r["sample_and_optimizer_seed"] == C.TUNING_SEED
                     and (m != C.CE or r.get("head_lr") == ce_head_lr)}
            missing = [lr for lr in C.LEARNING_RATES if lr not in cands]
            if missing:
                raise SystemExit(f"cannot select {m}, {k}-shot: no tuning run for lr={missing}"
                                 + (f" with head lr {ce_head_lr}" if m == C.CE and ce_head_lr else ""))
            best = sorted(cands, key=lambda lr: (-cands[lr], lr))[0]
            entry = {"lr": best, "val_macro_f1": cands[best],
                     "candidates": {f"{lr:g}": cands[lr] for lr in sorted(cands)}}
            if m == C.CE:
                entry["head_lr"] = ce_head_lr
            selected.setdefault(m, {})[str(k)] = entry
            log(f"locked {m}, {k}-shot: lr={best:g} (val macro-F1 "
                + ", ".join(f"{lr:g}: {v:.4f}" for lr, v in sorted(cands.items())) + ")")
    selected["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    write_json(path, selected)
    return selected


# ----------------------------------------------------------------------------- execution
def run_paths(root, spec: RunSpec):
    base = Path(root) / "results" / ("smoke" if spec.phase == "smoke" else "")
    return base / "runs" / f"{spec.run_id}.json", base / "preds"


def load_run_data(root, spec: RunSpec) -> RunData:
    val = load_split(root, "val")
    if spec.phase == "final":
        test = load_split(root, "test")
    elif spec.phase == "smoke":
        test = val.iloc[:300].copy()  # exercises the test path without touching test labels
    else:
        test = None
    return RunData(load_train(root, spec.k, spec.seed), val, test, load_labels(root))


def check_final_once(root, spec: RunSpec) -> None:
    runs_dir = Path(root) / "results" / "runs"
    pattern = f"{C.METHOD_SLUG[spec.method]}_k{spec.k}_seed{spec.seed}_*final.json"
    clash = [p.name for p in runs_dir.glob(pattern) if p.stem != spec.run_id]
    if clash:
        raise RuntimeError(f"{clash} already holds a test result for {spec.method}, {spec.k}-shot, "
                           f"seed {spec.seed} with other settings; each setting is scored once.")


def execute(spec: RunSpec, root, device, use_amp, encoder_factory, force=False):
    run_json, preds_dir = run_paths(root, spec)
    if run_json.exists() and not force:
        return "skipped", read_json(run_json)
    if spec.phase == "final":
        check_final_once(root, spec)
    data = load_run_data(root, spec)
    log(f"start {spec.run_id} (train {len(data.train)}, val {len(data.val)}, "
        f"test {'-' if data.test is None else len(data.test)})")
    result, preds = RUNNERS[spec.method](spec, data, encoder_factory, device, use_amp)
    preds_dir.mkdir(parents=True, exist_ok=True)
    for split, df in preds.items():
        df.to_csv(preds_dir / f"{spec.run_id}_{split}.csv", index=False)
    write_json(run_json, result)  # written last: its existence marks the run as complete
    return "done", result


def describe(r: dict) -> str:
    parts = [r["run_id"]]
    if r.get("best_step"):
        parts.append(f"best@{r['best_step']}")
    parts.append(f"val macro-F1 {r['val_macro_f1']:.4f}")
    if r.get("test_macro_f1") is not None:
        parts.append(f"TEST macro-F1 {r['test_macro_f1']:.4f}")
    parts.append(f"train {r['timing']['training_wall_sec_total']:.1f}s")
    if r.get("peak_gpu_memory_mb"):
        parts.append(f"peak {r['peak_gpu_memory_mb']:.0f} MB")
    return " | ".join(parts)


def save_environment(root, stage: str, device) -> None:
    env_dir = Path(root) / "results" / "env"
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    info = env_info(device)
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                text=True, timeout=120).stdout.splitlines()
    except Exception:  # noqa: BLE001 - best effort
        freeze = []
    write_json(env_dir / f"{stage}_{stamp}.json", {"env": info, "pip_freeze": freeze})


def main(argv=None, encoder_factory=None, expected=C.EXPECTED_COUNTS, verify_hash=True) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()

    if args.stage == "collect":
        report.collect(root)
        return 0
    if args.stage == "select":
        select_hparams(root, args.shots or C.SHOTS, args.ce_head_lr, args.force,
                       methods=_methods(args, (C.CE, C.SETFIT)))
        return 0

    prepare_data(root, expected=expected, force=args.force_data,
                 verify_hash=verify_hash and not args.allow_hash_mismatch)
    if args.stage == "data":
        return 0

    specs = plan(args.stage, args, root)
    device = get_device()
    use_amp = device.type == "cuda" and not args.no_amp
    if encoder_factory is None:
        def encoder_factory():
            return SentenceEncoder(device=device, use_amp=use_amp)
    info = env_info(device)
    log(f"device {info.get('gpu', device)} | torch {info['torch']} | transformers "
        f"{info['transformers']} | FP16 autocast {use_amp} | {len(specs)} planned runs")
    if device.type != "cuda":
        log("WARNING: no GPU found (Colab: Runtime > Change runtime type > T4 GPU)")

    if args.dry_run:
        for s in specs:
            done = run_paths(root, s)[0].exists()
            print(f"  [{'done' if done else 'todo'}] {s.run_id}")
        return 0

    save_environment(root, args.stage, device)
    failures = []
    for i, spec in enumerate(specs, 1):
        try:
            status, r = execute(spec, root, device, use_amp, encoder_factory, args.force)
            log(f"[{i}/{len(specs)}] {status}: {describe(r)}")
            if spec.phase == "smoke" and status == "done" and r.get("max_updates"):
                per = r["timing"]["encoder_update_sec"] / r["max_updates"]
                log(f"  {spec.method}: {per:.3f} s/update -> 8-shot updates ~"
                    f"{per * C.MAX_UPDATES[8] / 60:.1f} min, 32-shot ~{per * C.MAX_UPDATES[32] / 60:.1f} min "
                    "(plus checkpoint validation)")
        except Exception:  # noqa: BLE001 - keep going; report at the end
            failures.append(spec.run_id)
            traceback.print_exc()
            if args.fail_fast or spec.phase == "smoke":
                raise
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if spec.method == C.FROZEN and spec.k == min(C.SHOTS) and spec.seed == C.TUNING_SEED:
            report.lock_confusable_pairs(root)

    if args.stage == "tune":
        print()
        report.print_tuning_table(root)
        log("next: check the curves (--stage collect), then --stage select")
    if failures:
        log(f"{len(failures)} run(s) failed: {failures} - fix and re-run this stage")
        return 1
    log("stage finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
