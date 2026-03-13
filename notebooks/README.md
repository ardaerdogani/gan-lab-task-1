# Notebook Workflow

This folder contains two workflow styles.

## Manual Modular Workflow

Use this when you want step-by-step control:

1. `01_data_setup.ipynb`
2. `02_train_gan.ipynb`
3. `03_generate_synthetic_data.ipynb`
4. `04_classifier_experiments.ipynb`
5. `05_task1_results_and_analysis.ipynb`

## Optional Batch Workflow

Use this when you want the full Task 1 sweep across multiple dataset sizes:

1. `01_data_setup.ipynb`
2. `06_task1_batch_runner.ipynb`
3. `05_task1_results_and_analysis.ipynb`
4. `07_report_exports.ipynb`

Notes:

- `06_task1_batch_runner.ipynb` automates the same loop as `02 -> 03 -> 04`
- `07_report_exports.ipynb` handles report exports only
- the remaining `.py` files are shared config and model definitions
- open the notebooks from the repo root or from this folder; each notebook resets `cwd` to the project root
- generated outputs such as `runs/`, `reports/`, and `data_synth/` are local artifacts and are ignored by Git
