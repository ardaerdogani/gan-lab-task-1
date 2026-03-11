# Synthetic Data Generation with CGANs for Fruit Classification

This repository evaluates whether a conditional GAN can generate useful synthetic fruit images for downstream classification when real labeled data are limited.

The project is now notebook-first. Training, synthetic generation, Task 1 experiments, report exports, and output archiving are all driven from `notebooks/`.

## Notebook Workflow

Run the notebooks in this order:

1. `notebooks/01_data_setup.ipynb`
2. `notebooks/02_train_gan.ipynb`
3. `notebooks/03_generate_synthetic_data.ipynb`
4. `notebooks/04_classifier_experiments.ipynb`
5. `notebooks/05_task1_pipeline_and_results.ipynb`
6. `notebooks/06_report_exports_and_archiving.ipynb`

What each notebook covers:

- `01_data_setup.ipynb` prepares or inspects the dataset layout.
- `02_train_gan.ipynb` trains the CGAN and records FID over time.
- `03_generate_synthetic_data.ipynb` generates balanced synthetic image pools from a trained generator.
- `04_classifier_experiments.ipynb` runs standalone classifier baselines.
- `05_task1_pipeline_and_results.ipynb` runs the fair Task 1 pipeline end to end.
- `06_report_exports_and_archiving.ipynb` regenerates report tables, the FID SVG, and optional archives.

## Task 1 Coverage

The notebook pipeline is set up for the corrected Task 1 comparison:

- the GAN can be retrained separately for each data budget
- the synthetic pool is generated from the matching real-data budget
- `real`, `synth`, and `both` are evaluated without classical augmentation
- `real_aug` remains available as the optional non-generative baseline
- reported pipeline cost can include GAN training and synthetic image generation

## Repository Structure

```text
gan-lab/
├── config.py
├── models/
│   ├── gan.py
│   └── classifier.py
├── notebooks/
│   ├── 01_data_setup.ipynb
│   ├── 02_train_gan.ipynb
│   ├── 03_generate_synthetic_data.ipynb
│   ├── 04_classifier_experiments.ipynb
│   ├── 05_task1_pipeline_and_results.ipynb
│   ├── 06_report_exports_and_archiving.ipynb
│   └── README.md
├── data_final/
├── data_splits/
├── archives/
└── requirements.txt
```

Generated locally but not committed:

- `runs/`
- `data_synth/`
- `reports/`

## Dataset

The project uses a 3-class fruit dataset in `ImageFolder` format:

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

## Models

- GAN: conditional generator plus projection discriminator
- Classifier: compact CNN (`FruitCNN`)
- Image size: resized to `64x64` for model training

Important defaults live in `config.py`, including:

- latent dimension
- GAN epochs and batch size
- classifier epochs and batch size
- optimizer settings
- device selection

## Setup

Install dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- On Apple Silicon, the config prefers `mps` when available.
- For notebook use, install a notebook environment separately if needed, for example `jupyterlab` and `ipykernel`.

## Generated Outputs

Running the notebooks can create:

- `data_splits/task1/n*/train/...`
- `runs/task1/gan/n*/...`
- `runs/task1/synth/n*/...`
- `runs/task1/clf/all_results.json`
- `runs/task1/pipeline_summary.json`
- `runs/task1/clf/plots/*.png`
- `reports/tables/*.csv`
- `reports/figures/fid_vs_epoch.svg`

The standalone GAN notebook may also write:

- `runs/gan/train_log.json`
- `runs/gan/checkpoints/*.pt`
- `runs/gan/samples_epoch*.png`

## Remaining Python Files

The remaining `.py` files are shared code without notebook duplicates:

- `config.py`
- `models/gan.py`
- `models/classifier.py`

## Notes

- The correct notebook extension is `.ipynb`.
- Each notebook resets the working directory to the repo root.
- Generated outputs are intentionally ignored by Git to keep the repository clean between runs.
