# DML_v1

This repository contains DML multimodal learning experiments adapted to multiple datasets. Each dataset project keeps its own model and data-loading code, while training outputs follow the same practical style where possible: robustness/noise evaluation, a per-run `final_results.json`, and an aggregate `all_experiments.json`.

## Project Layout

| Project | Dataset(s) |
| --- | --- |
| `RGB_v1` | NYU Depth V2, SUN RGB-D |
| `CMU_v1` | CMU-MOSI, CMU-MOSEI |
| `CREMAD_v1` | CREMA-D |
| `Food_v1` | UPMC Food-101 |
| `MVSA` | MVSA-Single |

## Feature Matrix

| Dataset | Decision Fusion | Logit Fusion + Information Bottleneck | Logit Fusion + Conformal Prediction | Logit Fusion + Information Bottleneck + Conformal Prediction |
| --- | --- | --- | --- | --- |
| NYU Depth V2 | ✓ | ✓ |  |  |
| SUN RGB-D | ✓ | ✓ |  |  |
| CMU-MOSI | ✓ |  |  |  |
| CMU-MOSEI | ✓ |  |  |  |
| CREMA-D | ✓ |  |  |  |
| Food-101 | ✓ |  |  |  |
| MVSA-Single | ✓ | ✓ |  |  |

`✓` means the dataset currently has that training/evaluation variant implemented. Empty cells are planned or not yet implemented.

## Output Convention

Each training run writes its own artifacts under the dataset project's save directory. The expected core outputs are:

- `training.log`: run log.
- `model_best_clean.pt`: best checkpoint selected on clean evaluation.
- `final_results.json`: final clean and noisy/robustness results for the run.
- `all_experiments.json`: append-only summary of runs for the dataset.

Configuration is kept per project instead of using a shared framework. RGB, CMU, Food, and MVSA use Python CLI arguments; CREMAD uses its JSON/config dictionary path consistently.

## Run Projects Sequentially

Use `run_all_projects.bat` from the repository root to run the implemented project entrypoints in this order: `CREMAD_v1`, `Food_v1`, `MVSA`, `RGB_v1` NYU, and `RGB_v1` SUN.

Preview commands without starting training:

```bat
run_all_projects.bat --dry-run
```

Start the full sequence with the default PyTorch 2.5 environment:

```bat
run_all_projects.bat
```

Use another Python executable when needed:

```bat
run_all_projects.bat --python E:\anaconda3\envs\pytorch2.5\python.exe
```

The PowerShell version, `run_all_projects.ps1`, is also kept for users who prefer PowerShell and need per-project argument arrays.
