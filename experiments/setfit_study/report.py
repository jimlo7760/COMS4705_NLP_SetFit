"""Aggregate finished runs: results log, summaries, LaTeX table, curves, confusable pairs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .utils import log, read_json, write_json

# Column order of COMS4705_Milestone_Draft/results_log_template.csv, then extras.
TEMPLATE_COLUMNS = [
    "method", "shots_per_intent", "sample_and_optimizer_seed", "test_macro_f1", "test_accuracy",
    "training_wall_sec_total", "pair_build_sec", "encoder_update_sec",
    "checkpoint_train_embedding_sec", "checkpoint_lr_fit_sec", "checkpoint_validation_sec",
    "final_lr_fit_sec", "test_inference_sec_separate", "peak_gpu_memory_mb",
    "lr_checkpoint_fits_expected_max", "lr_checkpoint_fits_observed", "notes",
]
EXTRA_COLUMNS = [
    "phase", "run_id", "learning_rate", "head_lr", "best_step", "best_val_macro_f1",
    "val_macro_f1", "val_accuracy", "final_train_embedding_sec", "gpu", "amp_fp16",
    "truncation_train", "truncation_val", "truncation_test", "code_fingerprint", "created_utc",
]
METHOD_ORDER = {m: i for i, m in enumerate(C.METHODS)}

# Figure styling (dataviz reference palette, light mode; one hue per method, fixed order).
SERIES = {C.SETFIT: "#2a78d6", C.CE: "#eb6834"}
TEXT_PRIMARY, TEXT_SECONDARY, GRID, BASELINE = "#0b0b0b", "#52514e", "#e4e3df", "#52514e"


# ----------------------------------------------------------------------------- loading
def results_dir(root) -> Path:
    return Path(root) / "results"


def load_runs(res_dir, phases=("tuning", "final")) -> list:
    runs = [read_json(p) for p in sorted((Path(res_dir) / "runs").glob("*.json"))]
    return [r for r in runs if r.get("phase") in phases]


def runs_frame(runs: list) -> pd.DataFrame:
    rows = []
    for r in runs:
        t = r.get("timing", {})
        notes = []
        if r.get("phase") == "tuning":
            notes.append("tuning run: validation only")
        fc = r.get("final_classifier") or {}
        if fc and not fc.get("lr_converged", True):
            notes.append("final LR hit max_iter")
        if r.get("head_lr") is not None:
            notes.append(f"CE head lr {r['head_lr']:g}")
        tr = r.get("truncation_rate") or {}
        rows.append({
            "method": r["method"], "shots_per_intent": r["shots_per_intent"],
            "sample_and_optimizer_seed": r["sample_and_optimizer_seed"],
            "test_macro_f1": r.get("test_macro_f1"), "test_accuracy": r.get("test_accuracy"),
            **{k: t.get(k) for k in ("training_wall_sec_total", "pair_build_sec", "encoder_update_sec",
                                     "checkpoint_train_embedding_sec", "checkpoint_lr_fit_sec",
                                     "checkpoint_validation_sec", "final_lr_fit_sec",
                                     "test_inference_sec_separate", "final_train_embedding_sec")},
            "peak_gpu_memory_mb": r.get("peak_gpu_memory_mb"),
            "lr_checkpoint_fits_expected_max": r.get("lr_checkpoint_fits_expected_max"),
            "lr_checkpoint_fits_observed": r.get("lr_checkpoint_fits_observed"),
            "notes": "; ".join(notes),
            "phase": r["phase"], "run_id": r["run_id"], "learning_rate": r.get("learning_rate"),
            "head_lr": r.get("head_lr"), "best_step": r.get("best_step"),
            "best_val_macro_f1": r.get("best_val_macro_f1"), "val_macro_f1": r.get("val_macro_f1"),
            "val_accuracy": r.get("val_accuracy"), "gpu": (r.get("env") or {}).get("gpu"),
            "amp_fp16": r.get("amp_fp16"), "truncation_train": tr.get("train"),
            "truncation_val": tr.get("val"), "truncation_test": tr.get("test"),
            "code_fingerprint": (r.get("env") or {}).get("code_fingerprint"),
            "created_utc": r.get("created_utc"),
        })
    df = pd.DataFrame(rows, columns=TEMPLATE_COLUMNS + EXTRA_COLUMNS)
    if len(df):
        df = df.assign(_m=df["method"].map(METHOD_ORDER)).sort_values(
            ["phase", "shots_per_intent", "_m", "sample_and_optimizer_seed", "learning_rate"],
            na_position="first").drop(columns="_m")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- tables
def summarize_final(df: pd.DataFrame) -> pd.DataFrame:
    final = df[df["phase"] == "final"]
    if final.empty:
        return pd.DataFrame()
    g = final.groupby(["method", "shots_per_intent"])
    out = g.agg(n_seeds=("sample_and_optimizer_seed", "nunique"),
                test_macro_f1_mean=("test_macro_f1", "mean"), test_macro_f1_sd=("test_macro_f1", "std"),
                test_accuracy_mean=("test_accuracy", "mean"), test_accuracy_sd=("test_accuracy", "std"),
                train_wall_sec_mean=("training_wall_sec_total", "mean"),
                train_wall_sec_sd=("training_wall_sec_total", "std"),
                peak_gpu_memory_mb_mean=("peak_gpu_memory_mb", "mean")).reset_index()
    out["_m"] = out["method"].map(METHOD_ORDER)
    return out.sort_values(["shots_per_intent", "_m"]).drop(columns="_m").reset_index(drop=True)


def paired_differences(df: pd.DataFrame) -> pd.DataFrame:
    """Per seed (same training sample): A minus B on test macro-F1 and accuracy."""
    final = df[df["phase"] == "final"]
    rows = []
    for (k, seed), g in final.groupby(["shots_per_intent", "sample_and_optimizer_seed"]):
        by = g.set_index("method")
        for a, b in ((C.SETFIT, C.FROZEN), (C.SETFIT, C.CE), (C.CE, C.FROZEN)):
            if a in by.index and b in by.index:
                rows.append({"shots_per_intent": k, "seed": seed, "comparison": f"{a} - {b}",
                             "d_test_macro_f1": by.at[a, "test_macro_f1"] - by.at[b, "test_macro_f1"],
                             "d_test_accuracy": by.at[a, "test_accuracy"] - by.at[b, "test_accuracy"],
                             "train_wall_ratio": by.at[a, "training_wall_sec_total"]
                             / max(by.at[b, "training_wall_sec_total"], 1e-9)})
    return pd.DataFrame(rows)


def tuning_table(df: pd.DataFrame) -> pd.DataFrame:
    tun = df[df["phase"] == "tuning"]
    cols = ["method", "shots_per_intent", "learning_rate", "head_lr", "best_step",
            "best_val_macro_f1", "val_accuracy", "training_wall_sec_total"]
    return tun[cols].reset_index(drop=True)


def _fmt(mean, sd, digits=3):
    if mean is None or pd.isna(mean):
        return "--"
    if sd is None or pd.isna(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


def latex_main_table(summary: pd.DataFrame) -> str:
    """Drop-in body for Table 2 of the milestone (mean +- sample SD over final seeds)."""
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"& \multicolumn{2}{c}{Test macro-F1 $\uparrow$} & \multicolumn{2}{c}{Train wall time (s) $\downarrow$} \\",
             r"Method & 8-shot & 32-shot & 8-shot & 32-shot \\", r"\midrule"]
    for m in C.METHODS:
        cells_f1, cells_t = [], []
        for k in C.SHOTS:
            row = summary[(summary["method"] == m) & (summary["shots_per_intent"] == k)] if len(summary) else summary
            if len(row):
                r = row.iloc[0]
                cells_f1.append(_fmt(r["test_macro_f1_mean"], r["test_macro_f1_sd"]))
                cells_t.append(_fmt(r["train_wall_sec_mean"], r["train_wall_sec_sd"], 0))
            else:
                cells_f1.append("--")
                cells_t.append("--")
        lines.append(f"{m} & " + " & ".join(cells_f1 + cells_t) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- confusable pairs
def lock_confusable_pairs(root, n_pairs: int = 5):
    """Pre-select the 5 most confused intent pairs from seed-0 8-shot Frozen+LR *validation*
    predictions (ties alphabetical). Written once, before any comparison on test."""
    res = results_dir(root)
    out = res / "confusable_pairs.json"
    if out.exists():
        return read_json(out)
    src = res / "preds" / f"frozen_k{min(C.SHOTS)}_seed{C.TUNING_SEED}_tuning_val.csv"
    if not src.exists():
        return None
    df = pd.read_csv(src, keep_default_na=False)
    counts: dict = {}
    for t, p in zip(df["true_label"], df["pred_label"]):
        if t != p:
            key = tuple(sorted((t, p)))
            counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))[:n_pairs]
    obj = {"source": src.name,
           "rule": "largest C_ij + C_ji on seed-0 8-shot Frozen+LR validation; ties alphabetical",
           "pairs": [{"intent_a": a, "intent_b": b, "val_confusions": c} for (a, b), c in ranked]}
    write_json(out, obj)
    log("locked confusable pairs: " + "; ".join(f"{a} / {b} ({c})" for (a, b), c in ranked))
    return obj


def confusable_pair_rates(root) -> pd.DataFrame:
    """(C_ij + C_ji) / (n_i + n_j) on test for each locked pair and each final run."""
    res = results_dir(root)
    locked = res / "confusable_pairs.json"
    if not locked.exists():
        return pd.DataFrame()
    pairs = read_json(locked)["pairs"]
    rows = []
    for r in load_runs(res, phases=("final",)):
        path = res / "preds" / f"{r['run_id']}_test.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, keep_default_na=False)
        for p in pairs:
            a, b = p["intent_a"], p["intent_b"]
            n = int(df["true_label"].isin([a, b]).sum())
            c = int((((df["true_label"] == a) & (df["pred_label"] == b))
                     | ((df["true_label"] == b) & (df["pred_label"] == a))).sum())
            rows.append({"method": r["method"], "shots_per_intent": r["shots_per_intent"],
                         "seed": r["sample_and_optimizer_seed"], "pair": f"{a} / {b}",
                         "confusions": c, "gold_in_pair": n, "rate": c / n if n else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def plot_training_curves(runs: list, out_dir: Path) -> list:
    """Seed-0 tuning runs: validation macro-F1 per checkpoint (Frozen+LR as reference),
    CE loss and SetFit pair loss, one row per training size. One y-scale per panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tuning = [r for r in runs if r.get("phase") == "tuning"]
    shots = sorted({r["shots_per_intent"] for r in tuning if r["method"] != C.FROZEN})
    if not shots:
        return []
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, axes = plt.subplots(len(shots), 3, figsize=(7.0, 2.25 * len(shots)), squeeze=False,
                             constrained_layout=True)
    styles = ["-", "--", ":", "-."]
    for row, k in enumerate(shots):
        ax_f1, ax_ce, ax_sf = axes[row]
        variants = {}
        for r in sorted((r for r in tuning if r["shots_per_intent"] == k),
                        key=lambda r: (METHOD_ORDER[r["method"]], r.get("head_lr") or 0,
                                       -(r.get("learning_rate") or 0))):
            if r["method"] == C.FROZEN:
                ax_f1.axhline(r["val_macro_f1"], color=BASELINE, lw=1.0, ls=(0, (1, 2)))
                ax_f1.annotate("Frozen+LR", xy=(0.0, r["val_macro_f1"]), xycoords=("axes fraction", "data"),
                               xytext=(3, 3), textcoords="offset points", ha="left",
                               color=TEXT_SECONDARY, fontsize=7)
                continue
            i = variants.setdefault(r["method"], 0)
            variants[r["method"]] += 1
            label = f"{r['method']}, lr {r['learning_rate']:g}"
            if r.get("head_lr") is not None:
                label += f", head {r['head_lr']:g}"
            kw = dict(color=SERIES[r["method"]], ls=styles[i % len(styles)], lw=1.5)
            h = r.get("history") or []
            ax_f1.plot([e["step"] for e in h], [e["val_macro_f1"] for e in h], marker="o", ms=3.5,
                       label=label, **kw)
            lc = np.asarray(r.get("loss_curve") or [[np.nan, np.nan]], dtype=float)
            (ax_ce if r["method"] == C.CE else ax_sf).plot(lc[:, 0], lc[:, 1], label=label, **kw)
        ax_f1.set_title(f"{k}-shot: val. macro-F1", color=TEXT_PRIMARY, loc="left")
        ax_ce.set_title(f"{k}-shot: CE loss", color=TEXT_PRIMARY, loc="left")
        ax_sf.set_title(f"{k}-shot: SetFit pair loss", color=TEXT_PRIMARY, loc="left")
        for ax in (ax_f1, ax_ce, ax_sf):
            ax.set_xlabel("encoder updates", color=TEXT_SECONDARY)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=TEXT_SECONDARY)
    # one legend for the figure: colour = method, line style = learning rate, in every panel
    handles, labels = {}, []
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in handles:
                handles[lab] = h
                labels.append(lab)
    if labels:
        fig.legend([handles[lab] for lab in labels], labels, loc="outside upper center",
                   ncol=min(4, len(labels)), frameon=False, labelcolor=TEXT_SECONDARY)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "tuning_curves.pdf", out_dir / "tuning_curves.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=200)
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------- entry points
def print_tuning_table(root) -> pd.DataFrame:
    df = runs_frame(load_runs(results_dir(root), phases=("tuning",)))
    tab = tuning_table(df)
    if len(tab):
        with pd.option_context("display.width", 140, "display.max_columns", 20):
            print(tab.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return tab


def collect(root) -> dict:
    res = results_dir(root)
    runs = load_runs(res)
    if not runs:
        log("no finished runs yet")
        return {}
    df = runs_frame(runs)
    df.to_csv(res / "results_log.csv", index=False)
    summary = summarize_final(df)
    paired = paired_differences(df)
    tuning = tuning_table(df)
    pairs = confusable_pair_rates(root)
    summary.to_csv(res / "summary.csv", index=False)
    paired.to_csv(res / "paired_differences.csv", index=False)
    tuning.to_csv(res / "tuning_summary.csv", index=False)
    if len(pairs):
        pairs.to_csv(res / "confusable_pairs_test.csv", index=False)
    (res / "table_main.tex").write_text(latex_main_table(summary))
    figs = plot_training_curves(runs, res / "figures")

    fmt = {"float_format": lambda x: f"{x:.4f}"}
    with pd.option_context("display.width", 160, "display.max_columns", 30):
        if len(tuning):
            print("\nTuning (seed 0, validation only):\n" + tuning.to_string(index=False, **fmt))
        if len(summary):
            print("\nFinal runs (test; mean and sample SD over seeds):\n" + summary.to_string(index=False, **fmt))
        if len(paired):
            agg = paired.groupby(["shots_per_intent", "comparison"])["d_test_macro_f1"].agg(["mean", "std", "count"])
            print("\nPaired test macro-F1 differences:\n" + agg.to_string(**fmt))
    log(f"wrote {res/'results_log.csv'}, summary.csv, paired_differences.csv, table_main.tex"
        + (f", {figs[0].name}" if figs else ""))
    return {"runs": df, "summary": summary, "paired": paired, "tuning": tuning, "pairs": pairs}
