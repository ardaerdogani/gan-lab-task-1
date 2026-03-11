# Notebook Workflow

This folder is the primary workflow for the project.

Run notebooks in order:

1. `01_data_setup.ipynb`
2. `02_train_gan.ipynb`
3. `03_generate_synthetic_data.ipynb`
4. `04_classifier_experiments.ipynb`
5. `05_task1_pipeline_and_results.ipynb`
6. `06_report_exports_and_archiving.ipynb`

What changed:

- training and experiment entrypoints live in notebooks
- report exports and archiving now live in notebooks too
- the remaining `.py` files are only shared config and model definitions

Notes:

- The correct extension is `.ipynb`
- Open the notebooks from the repo root or from this folder; each notebook resets `cwd` to the project root
- Generated outputs such as `runs/`, `reports/`, and `data_synth/` are local artifacts and are ignored by Git
