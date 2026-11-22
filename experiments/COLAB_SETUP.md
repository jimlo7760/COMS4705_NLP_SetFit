# Running the experiments on Colab

Everything the proposal specifies (data audit, splits, Frozen+LR, CE tuning, SetFit,
seed-0 learning-rate selection, final seeds 11/22/33, timing, confusable-pair lock) is
implemented in `setfit_study/` and driven by one notebook, `colab_run.ipynb`.

Using Google Cloud credits on a Compute Engine VM instead? Follow `GCP_VM_SETUP.md`.

## What is in this folder

| Path | What it is |
|---|---|
| `colab_run.ipynb` | The notebook you run on Colab, top to bottom |
| `setfit_study/config.py` | Every protocol setting in one place (seeds, sizes, learning rates, caps) |
| `setfit_study/data.py` | Download, audit, validation split, nested 8/32-shot samples |
| `setfit_study/pairs.py` | SetFit pair sampler (4 positive + 4 negative partners per anchor) |
| `setfit_study/models.py` | all-MiniLM-L6-v2 with mean pooling and L2 normalisation |
| `setfit_study/train_frozen.py`, `train_ce.py`, `train_setfit.py` | The three methods |
| `setfit_study/run_all.py` | Stages, resume logic, learning-rate lock |
| `setfit_study/report.py` | Results log, summary tables, LaTeX table, curves, confusable pairs |
| `tests/test_pipeline.py` | Offline tests (synthetic data, stand-in encoder) |

## One-time setup (about 5 minutes)

1. In Google Drive, create a folder named `coms4705_setfit` directly under **My Drive**.
2. Upload this whole `experiments/` folder into it (drag the folder onto drive.google.com).
   You should end up with `My Drive/coms4705_setfit/experiments/setfit_study/...`.
   If you use another folder name, change `ROOT` in the notebook's first code cell.
3. In Drive, open `experiments/colab_run.ipynb` → *Open with → Google Colaboratory*.
4. **Runtime → Change runtime type → T4 GPU.** T4 is the cheapest GPU and matches the
   proposal's hardware. The model is small (22.7M parameters), so an A100 mostly burns
   compute units faster.

## Every session

Run the three setup cells (they mount Drive, check the GPU, and `cd` into the code),
then the stage you are on. Drive authorisation pops up once per session.

## Run order

| Step | Stage | What happens | Output (under `results/`) |
|---|---|---|---|
| 1 | `smoke` | Downloads and audits the data (stops on any mismatch), builds the splits, then 10 updates of CE and SetFit plus Frozen+LR; prints seconds per update | `../data/`, `smoke/` (ignored later) |
| 2 | `frozen` | Frozen+LR for 8/32-shot, seeds 0/11/22/33 | `runs/`, `preds/`, `confusable_pairs.json` |
| 3 | `tune` | CE and SetFit, seed 0, lr 1e-5 and 3e-5 (8 runs, validation only) | `runs/*_tuning.json` |
| 3b | `collect` | **Look at the curves** before locking (see "CE check" below) | `figures/tuning_curves.png` |
| 4 | `select` | Locks one lr per method and size | `selected_hparams.json` |
| 5 | `final` | CE and SetFit on seeds 11/22/33, each scored once on test (12 runs) | `runs/*_final.json`, `preds/*_test.csv` |
| 6 | `collect` | All tables and the figure for the report | see below |

Do 8-shot first at every step (`--shots 8`); the notebook already splits it that way.
That alone gives you a complete 8-shot comparison for the milestone.

**Time.** I have not timed this on a GPU. The smoke stage prints seconds per update for
each method; multiply by 300 (8-shot) or 1,000 (32-shot) updates and add the checkpoint
validations. For a model this size I'd expect minutes per run on a T4, not hours.

## Protocol guardrails built in

- Tuning runs (seed 0) never load the test set; final runs score it exactly once.
  A second final run for the same method/size/seed with different settings is refused.
- `selected_hparams.json` is written once per method and size; overwriting needs `--force`.
- The data stage recomputes the proposal's audit numbers and file hashes and stops on a
  mismatch. Don't edit the expected numbers to get past it; find out what changed.
- The five confusable intent pairs are locked from seed-0 8-shot **validation**
  predictions, before any test comparison.
- Every run records GPU, library versions, model commit and a hash of the code;
  each stage saves `pip freeze` to `results/env/` (your environment lockfile).

## CE check (do this after tuning)

The CE head is a new, randomly initialised 77-way layer on unit-length sentence vectors,
trained with encoder-sized learning rates (1e-5 / 3e-5). It may learn very slowly. In the
offline test with a stand-in encoder, CE at 3e-5 stayed at chance-level loss (≈ ln 77 ≈ 4.3)
for all 300 updates. If the CE loss panel looks like that on the real model too, CE would
be a broken baseline, not a fair one. Fix it by tuning CE with a separate head learning rate:

```
--stage tune --methods ce --ce-head-lr 1e-3
--stage select --methods ce --ce-head-lr 1e-3
```

Report it in the milestone as a change since the proposal, with the curve as the reason.

## Saving credits

- **Runtime → Disconnect and delete runtime** as soon as a stage finishes. A connected
  GPU runtime keeps using compute units while idle.
- Keep the browser tab open while a stage runs. If Colab disconnects anyway, reconnect,
  re-run the setup cells and the same stage: finished runs are skipped.
- To split work across sessions or teammates, use `--seeds 11`, `--seeds 22`, ... in
  separate sessions. Never run the *same* runs in two sessions at once.

## What to bring back for the milestone

From `results/`:
- `table_main.tex`: drop-in body for Table 2 (test macro-F1 and training time,
  mean ± sample SD over seeds 11/22/33)
- `summary.csv`, `paired_differences.csv` (per-seed SetFit − baseline differences)
- `tuning_summary.csv` and `figures/tuning_curves.pdf` (training curves; figures do not
  count toward the 2-page limit)
- `results_log.csv`: one row per run, same columns as `results_log_template.csv`
- `confusable_pairs.json`, `confusable_pairs_test.csv`: the seed-wise pair analysis
- `preds/`: per-example predictions for the error analysis later

## If something fails

- `No GPU`: set the runtime type to T4 GPU and re-run the setup cells.
- `data audit does not match the proposal`: stop and look at the printed counts.
- `SHA-256 ... differs`: the upstream CSV changed. Compare the audit counts; continue with
  `--allow-hash-mismatch` only if they match, and mention it in the report.
- A run crashes: the stage continues with the other runs and lists the failures at the end;
  fix the cause and run the same stage again.

## Running the offline tests (optional, on a laptop)

```
cd experiments
python -m pytest -q tests
```

These use synthetic data and a tiny stand-in encoder, so they need no downloads. They do
**not** exercise the Hugging Face encoder or the GPU/FP16 path; the Colab smoke stage does.
