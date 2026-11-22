# Few-shot BANKING77 intent classification: a controlled study of SetFit

COMS 4705 (Fall 2026) final project.

| Folder | Contents |
| --- | --- |
| [`proposal/`](proposal/) | Submitted proposal (`main.tex`, `main.pdf`), the earlier placeholder draft (`SetFit_BANKING77_Proposal.tex`), the course template (`template.tex`), the data audit (`data_audit.json`) and the Chinese notes (`README_CN.md`, `SetFit_BANKING77_Chinese_Guide.md`). |
| [`milestone/`](milestone/) | Milestone report (`COMS4705_Milestone_Report.tex`, `.pdf`), its figure, bibliography, the course milestone template and `results_log_template.csv`. |
| [`experiments/`](experiments/) | The `setfit_study` pipeline, tests, the Colab notebook and the Colab and GCP VM guides. |
| [`data/`](data/) | The raw BANKING77 CSVs and the cleaned validation, test and few-shot training splits the pipeline built from them. See [Data](#data). |
| [`results/`](results/) | Output of the full pipeline run (per-run records, predictions, summary tables, the LaTeX results table and the tuning-curve figure). See [Results](#results). |

The pipeline writes `data/` and `results/` next to `experiments/`, so both folders stay at the
top level.

Each LaTeX folder compiles on its own with pdfLaTeX: open the folder in Overleaf, or run
`pdflatex` / `bibtex` inside it.

To run the experiments, see [`experiments/COLAB_SETUP.md`](experiments/COLAB_SETUP.md) or
[`experiments/GCP_VM_SETUP.md`](experiments/GCP_VM_SETUP.md).

## Data

Built by the `data` stage of `setfit_study.run_all` from the PolyAI BANKING77 release
(77 intents). The protocol, source URLs and SHA-256 hashes of every file are recorded in
[`data/split_manifest.json`](data/split_manifest.json); the pipeline checks these hashes before it
reuses the folder.

Cleaning: strip outer whitespace, drop 4 redundant training rows, and drop the 6 training texts
that also occur in the test set (listed in the manifest). That leaves 9,993 clean training rows.
The validation set is 10% of each class (seed 2026); the remaining 9,028 rows form the pool the
few-shot samples are drawn from.

| File | Rows | Contents |
| --- | ---: | --- |
| `raw/train.csv`, `raw/test.csv` | 10,003 / 3,080 | Original BANKING77 files, columns `text,category`. |
| `label_map.json` | 77 | Intent names; a row's `label_id` is its index in this list. |
| `val.csv` | 965 | Validation split, used for tuning and checkpoint selection. |
| `test.csv` | 3,080 | Full original test set, scored once per final run. |
| `train_k8_seed{0,11,22,33}.csv` | 616 each | 8 examples per intent. |
| `train_k32_seed{0,11,22,33}.csv` | 2,464 each | 32 examples per intent. For each seed the 8-shot sample is nested inside the 32-shot one. |
| `split_manifest.json` | | Protocol, audit counts, removed overlap texts and file hashes. |

The cleaned CSVs share the columns `row_id,text,label,label_id`. Seed 0 is used only for tuning on
validation; seeds 11, 22 and 33 are the final runs.

## Results

All runs use `sentence-transformers/all-MiniLM-L6-v2` on one NVIDIA L4 GPU with FP16 autocast.
Learning rates were tuned on seed 0 (validation only) and then locked
([`results/selected_hparams.json`](results/selected_hparams.json)): CE tuning uses 3e-5 with a
head learning rate of 1e-3, and SetFit uses 1e-5, for both shot counts.

Test results, mean ± SD over seeds 11, 22 and 33
(from [`results/summary.csv`](results/summary.csv)):

| Method | Macro-F1, 8-shot | Macro-F1, 32-shot | Accuracy, 8-shot | Accuracy, 32-shot | Train time (s), 8-shot | Train time (s), 32-shot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen+LR | **0.788 ± 0.003** | **0.873 ± 0.002** | **0.798** | **0.874** | 0.7 | 1.7 |
| CE tuning | 0.706 ± 0.015 | 0.836 ± 0.002 | 0.724 | 0.841 | 12.2 | 37.6 |
| SetFit | 0.786 ± 0.002 | 0.871 ± 0.002 | 0.796 | 0.873 | 18.0 | 63.6 |

On every seed, SetFit beats CE tuning by 3.3 to 9.4 macro-F1 points, but finishes 0.03 to 0.28
points below Frozen+LR while training 16 to 50 times longer
([`results/paired_differences.csv`](results/paired_differences.csv)). The same table in LaTeX is
[`results/table_main.tex`](results/table_main.tex).

| Path | Contents |
| --- | --- |
| `results_log.csv` | One row per final run: test and validation scores, timing breakdown, peak GPU memory, environment. |
| `summary.csv` | Mean and SD per method and shot count (the table above). |
| `paired_differences.csv` | Per-seed differences in test macro-F1 and accuracy between methods, and the ratio of their training times. |
| `tuning_summary.csv` | Seed-0 validation scores for every learning-rate candidate. |
| `selected_hparams.json` | The locked learning rates and the selection rule. |
| `table_main.tex` | Main results table for the report. |
| `figures/tuning_curves.{pdf,png}` | Validation macro-F1 and training loss (CE loss, SetFit pair loss) over encoder updates for the tuning runs. |
| `confusable_pairs.json` | The five most-confused intent pairs, locked from seed-0 8-shot Frozen+LR validation predictions. |
| `confusable_pairs_test.csv` | Test confusion rate on those pairs for each final run. |
| `error_analysis/seed11_k8_correctness_sample.csv` | 40 seed-11 8-shot test examples, 10 from each cell: both right, both wrong, SetFit fixes a Frozen+LR error, SetFit breaks a Frozen+LR success. |
| `runs/<run_id>.json` | Full record of each run: config, environment, scores, per-class recall and training curve. |
| `preds/<run_id>_{val,test}.csv` | Per-example predictions (`row_id,true_id,pred_id,true_label,pred_label`). |
| `env/<stage>_<time>.json` | Python, CUDA and GPU details and `pip freeze` for each pipeline stage. |
| `smoke/` | Records and predictions from the 10-update smoke test; not used in the results. |

Run IDs read `<method>_k<shots>_seed<seed>[_lr<lr>][_hlr<head lr>]_<phase>`, where the phase is
`tuning` (validation only) or `final` (scored once on test). The stage logs in `results/logs/`
are not tracked in Git, since `*.log` is ignored.
