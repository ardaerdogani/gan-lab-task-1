# Synthetic Data Generation with CGANs for Fruit Classification

This repository evaluates whether a conditional GAN can generate useful synthetic fruit images for downstream classification when real labeled data are limited.

The project is notebook-first, and the workflow now runs through four sequential notebooks.

## Notebook Workflows

Use this when you want step-by-step control:

1. `notebooks/01_data_setup_and_train_gan.ipynb`
2. `notebooks/02_generate_synthetic_data_and_classifier_experiments.ipynb`
3. `notebooks/03_task1_results_and_analysis.ipynb`
4. `notebooks/04_report_exports.ipynb`

What each notebook does:

- `01_data_setup_and_train_gan.ipynb`: inspect dataset structure, preview real images, and train one GAN run
- `02_generate_synthetic_data_and_classifier_experiments.ipynb`: generate one synthetic dataset from a checkpoint and then run one classifier setup or a small grid
- `03_task1_results_and_analysis.ipynb`: analyze saved results and generate plots
- `04_report_exports.ipynb`: export report tables and generate the FID SVG figure

Notes:

- `04_report_exports.ipynb` is optional if you only need analysis outputs and not export artifacts

## Task 1 Coverage

The Task 1 workflow supports:

- retraining the GAN separately for each data budget
- generating the synthetic pool from the matching real-data budget
- evaluating `real`, `synth`, and `both` without classical augmentation
- keeping `real_aug` as the optional non-generative baseline
- tracking classifier-only time and full pipeline time

## Repository Structure

```text
gan-lab/
├── config.py
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
├── data_splits/
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

- `Config` now resolves device automatically in the order `cuda -> mps -> cpu`
- for notebook use, install a notebook environment separately if needed, for example `jupyterlab` and `ipykernel`

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

- The correct notebook extension is `.ipynb`
- Each notebook resets the working directory to the repo root
- Generated outputs are intentionally ignored by Git to keep the repository clean between runs
