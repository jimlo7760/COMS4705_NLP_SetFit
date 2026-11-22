# Few-shot BANKING77 intent classification: a controlled study of SetFit

COMS 4705 (Fall 2026) final project.

| Folder | Contents |
| --- | --- |
| [`proposal/`](proposal/) | Submitted proposal (`main.tex`, `main.pdf`), the earlier placeholder draft (`SetFit_BANKING77_Proposal.tex`), the course template (`template.tex`), the data audit (`data_audit.json`) and the Chinese notes (`README_CN.md`, `SetFit_BANKING77_Chinese_Guide.md`). |
| [`milestone/`](milestone/) | Milestone report (`COMS4705_Milestone_Report.tex`, `.pdf`), its figure, bibliography, the course milestone template and `results_log_template.csv`. |
| [`experiments/`](experiments/) | The `setfit_study` pipeline, tests, the Colab notebook and the Colab and GCP VM guides. |
| `results/` | Pipeline output (not in Git). The pipeline writes `data/` and `results/` next to `experiments/`, so this folder stays at the top level. |

Each LaTeX folder compiles on its own with pdfLaTeX: open the folder in Overleaf, or run
`pdflatex` / `bibtex` inside it.

To run the experiments, see [`experiments/COLAB_SETUP.md`](experiments/COLAB_SETUP.md) or
[`experiments/GCP_VM_SETUP.md`](experiments/GCP_VM_SETUP.md).
