# Synthetic Data Generation with CGANs for Fruit Classification

This repository studies whether a conditional GAN can generate useful synthetic fruit images for downstream classification when real labeled data are limited.

The project stays notebook-first, but the runtime logic now lives in `workflow.py`. The notebooks are thin entrypoints over the shared training, generation, evaluation, and Task 1 pipeline code.

## Notebook Workflow

Use the notebooks in this order:

1. `notebooks/01_data_setup_and_train_gan.ipynb`
2. `notebooks/02_generate_synthetic_data_and_classifier_experiments.ipynb`
3. `notebooks/03_task1_results_and_analysis.ipynb`
4. `notebooks/04_report_exports.ipynb`

What each notebook does:

- `01_data_setup_and_train_gan.ipynb`: dataset sanity checks, sample previews, and one standalone GAN run in `runs/gan/`
- `02_generate_synthetic_data_and_classifier_experiments.ipynb`: the authoritative Task 1 runner that writes standardized outputs to `runs/task1/`
- `03_task1_results_and_analysis.ipynb`: loads `runs/task1/` outputs and generates summary tables and plots
- `04_report_exports.ipynb`: exports CSV tables and optional SVG figures from the saved Task 1 outputs

## Runtime Defaults

`Config` keeps notebook defaults conservative, but the runtime is now aware of shared CUDA hosts:

- `device="auto"` prefers CUDA, then MPS, then CPU
- On multi-GPU CUDA systems, `device="auto"` selects the visible GPU with the most free memory
- `cuda_min_free_gib=2.0` fails early if the selected CUDA device is already too full for a new stage
- `pin_memory` is auto-enabled on CUDA unless explicitly overridden
- `allow_tf32=True` enables TensorFloat-32 math on CUDA for faster H100/A100-class runs
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by default unless you already exported your own allocator config
- `runtime_profile="m4_balanced"` is only applied automatically on Apple silicon
- `gan_batch=80`
- `clf_batch=64`
- `sample_every=25`
- `fid_every=25`
- `classifier_compile=False`

FID is enabled by default, uses the `train` split as the reference split, and validates the reference sample count before long runs start.

## Task 1 Outputs

The shared workflow writes Task 1 artifacts to:

- `runs/task1/gan/n*/...`
- `runs/task1/synth/n*/...`
- `runs/task1/clf/all_results.json`
- `runs/task1/pipeline_summary.json`
- `runs/task1/clf/plots/*.png`

Standalone GAN runs from notebook 01 still write to:

- `runs/gan/train_log.json`
- `runs/gan/checkpoints/*.pt`
- `runs/gan/samples_epoch*.png`

## Repository Structure

```text
gan-lab-task-1/
├── config.py
├── workflow.py
├── models/
│   ├── gan.py
│   └── classifier.py
├── notebooks/
│   ├── 01_data_setup_and_train_gan.ipynb
│   ├── 02_generate_synthetic_data_and_classifier_experiments.ipynb
│   ├── 03_task1_results_and_analysis.ipynb
│   ├── 04_report_exports.ipynb
│   └── README.md
├── data_final/
└── requirements.txt
```

Generated artifacts are local-only:

- `runs/`
- `reports/`
- `data_synth/`

## Dataset

The dataset is a 3-class `ImageFolder` tree:

```text
data_final/
├── train/
├── val/
└── test/
```

Classes:

- `apple`
- `banana`
- `orange`

## Setup

### Current H100 / CUDA 12.7 System

For the current shared system with NVIDIA H100 NVL GPUs and driver `565.57.01`, install a CUDA-enabled PyTorch build first, then the base project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name gan-lab-task-1 --display-name "Python (gan-lab-task-1)"
```

Why `cu126`: it is a good fit for a CUDA 12.7 driver stack while staying on the official current PyTorch wheel path.

On this host, leaving `device="auto"` is usually enough. If you want to pin a specific GPU from the notebooks, set:

```python
cfg = cfg.with_overrides(device="cuda:0")
```

or

```python
cfg = cfg.with_overrides(device="cuda:1")
```

If you hit a shared-GPU OOM on a pinned device, switch back to `device="auto"` or choose the roomier GPU, then restart the notebook kernel before rerunning the training cell.

If `device="auto"` still blocks on the shared-memory guard, you can lower the notebook override:

```python
CUDA_MIN_FREE_GIB_OVERRIDE = 1.5
```

Use that cautiously: it lowers the start threshold and may let a crowded GPU run, but it also increases the chance of a later CUDA OOM.

### Generic Install

Install PyTorch for your platform using the matching official command first, then install the base dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For notebook use, install a notebook environment if needed, for example `jupyterlab` and `ipykernel`.

## Notes

- Each notebook resets the working directory to the repo root.
- The notebooks now depend on the shared `workflow.py` module instead of redefining training loops inline.
- Notebook metadata now uses the generic `python3` kernel name so a local `ipykernel` works without a machine-specific `gan` kernel.
- Dependency checks happen early, so missing packages like `scikit-learn` fail in preflight instead of halfway through a long run.
