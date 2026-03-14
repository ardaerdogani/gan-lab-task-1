# Notebook Workflow

This folder contains the notebook workflow.

Use this when you want step-by-step control:

1. `01_data_setup_and_train_gan.ipynb`
2. `02_generate_synthetic_data_and_classifier_experiments.ipynb`
3. `03_task1_results_and_analysis.ipynb`
4. `04_report_exports.ipynb`

Notes:

- `01_data_setup_and_train_gan.ipynb` covers both dataset setup and GAN training
- `02_generate_synthetic_data_and_classifier_experiments.ipynb` covers both synthetic generation and classifier experiments
- `04_report_exports.ipynb` handles report exports only
- the remaining `.py` files are shared config and model definitions
- open the notebooks from the repo root or from this folder; each notebook resets `cwd` to the project root
- generated outputs such as `runs/`, `reports/`, and `data_synth/` are local artifacts and are ignored by Git
