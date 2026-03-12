# Notebook Workflow

This folder gives the project a notebook-first entry point without replacing the core Python code.

Order:

1. `01_data_setup.ipynb`
2. `02_train_gan.ipynb`
3. `03_generate_synthetic_data.ipynb`
4. `04_classifier_experiments.ipynb`
5. `05_task1_pipeline_and_results.ipynb`

Why this layout:

- notebooks are better for explanation, inline plots, and report-ready exploration
- `.py` files remain the source of truth for training logic and reproducibility
- the notebooks import or call the existing scripts instead of re-implementing them

Notes:

- The correct extension is `.ipynb`
- Open the notebooks from the repo root or from this folder; each notebook resets `cwd` to the project root
- If you want to run them locally, install a notebook environment separately, for example `pip install jupyterlab ipykernel`
