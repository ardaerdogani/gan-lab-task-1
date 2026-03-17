# Notebook Workflow

The notebooks are the user-facing entrypoints, but the training and evaluation code now lives in `workflow.py`.

Run them in this order:

1. `01_data_setup_and_train_gan.ipynb`
2. `02_generate_synthetic_data_and_classifier_experiments.ipynb`
3. `03_task1_results_and_analysis.ipynb`
4. `04_report_exports.ipynb`

Notebook roles:

- `01_data_setup_and_train_gan.ipynb`: dataset checks, sample previews, and one standalone GAN run in `runs/gan/`
- `02_generate_synthetic_data_and_classifier_experiments.ipynb`: the full Task 1 runner that writes standardized outputs to `runs/task1/`
- `03_task1_results_and_analysis.ipynb`: analysis and plot generation for Task 1 outputs
- `04_report_exports.ipynb`: CSV and SVG exports from the saved Task 1 outputs

Notes:

- Open notebooks either from the repo root or from this folder; each notebook resets `cwd` to the project root.
- For the current H100 / CUDA 12.7 system, install PyTorch with the official `cu126` wheels before `pip install -r requirements.txt`.
- The notebook setup cells expose a `DEVICE_OVERRIDE` value if you want to pin `cuda:0` or `cuda:1`; otherwise `device="auto"` selects the visible GPU with the most free memory.
- Generated artifacts such as `runs/`, `reports/`, and `data_synth/` are local outputs.
- If notebook 03 reports missing outputs, run notebook 02 first.
