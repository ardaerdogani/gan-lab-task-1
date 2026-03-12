# Synthetic Data Generation with CGANs for Fruit Classification

This repository studies whether a **Conditional GAN (CGAN)** can generate useful synthetic fruit images for downstream classification when real labeled data are limited.

The project trains a class-conditional generator on a 3-class fruit dataset (`apple`, `banana`, `orange`), evaluates generative quality with **FID**, builds balanced synthetic image pools, and measures downstream utility with Task 1-compatible classifier baselines:

1. `real` - train on real images only
2. `synth` - train on synthetic images only
3. `both` - train on real + synthetic images together
4. `real_aug` - optional real-only baseline with classical non-generative augmentation

The corrected Task 1 pipeline now supports **fair per-size evaluation**: for each training size `N`, the GAN can be retrained on exactly `N` real images per class before generating the synthetic pool used in downstream experiments.

The full thesis-style write-up is available at [reports/report.md](reports/report.md).

## Project Snapshot

- **Dataset:** Fruits-360-like subset with 3 classes
- **Image resolution:** raw 100x100, resized to 64x64 for training
- **GAN:** Conditional batch-normalized generator + projection discriminator
- **Classifier:** compact CNN (`FruitCNN`)
- **Best recorded FID:** `143.36` at epoch `140`
- **Best downstream result:** `99.66%` test accuracy at `400` images/class using `real + synthetic`

## Key Findings

- The CGAN learns a usable class-conditional distribution and improves substantially during training according to FID.
- Synthetic data can help as an **augmentation source**, but the gains are modest because real-only accuracy is already very high on this dataset.
- Synthetic-only training does **not** transfer well to real test data, which indicates a meaningful synthetic-to-real domain gap.
- FID is useful for monitoring training, but it is **not sufficient** as a stand-alone success metric. Downstream evaluation is necessary.

## Results

### FID During GAN Training

![FID vs Epoch](reports/figures/fid_vs_epoch.svg)

### Classification Accuracy vs Training Size

![Accuracy vs Data Size](reports/figures/accuracy_vs_size.png)

### Accuracy Summary

| Images per class | Real only | Synth only | Real + Synth |
|---:|---:|---:|---:|
| 100 | 98.58% | 93.16% | 98.98% |
| 200 | 99.25% | 90.72% | 98.78% |
| 400 | 98.37% | 85.37% | 99.66% |
| 800 | 99.93% | 89.43% | 99.93% |
| 1300 | 99.93% | 69.58% | 99.86% |

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
│   └── 05_task1_pipeline_and_results.ipynb
├── scripts/
│   ├── export_report_tables.py
│   └── plot_fid_svg.py
├── data_final/
├── data_splits/
├── reports/
│   ├── report.md
│   ├── figures/
│   └── tables/
└── requirements.txt
```

## Dataset

The dataset is arranged in `ImageFolder` format:

```text
data_final/
├── train/
├── val/
└── test/
```

Each split contains the same three classes:

- `apple`
- `banana`
- `orange`

Current split sizes:

| Split | Apple | Banana | Orange | Total |
|---:|---:|---:|---:|---:|
| train | 1300 | 1300 | 1300 | 3900 |
| val | 159 | 159 | 159 | 477 |
| test | 492 | 492 | 492 | 1476 |

## Method Overview

### 1. Train the CGAN

The generator maps a latent vector `z` and class label `y` to a `64x64` RGB image. The discriminator uses projection conditioning and is trained with `BCEWithLogitsLoss`.

Important defaults from `config.py`:

- `z_dim = 128`
- `gan_epochs = 200`
- `gan_batch = 96`
- `gan_lr_g = 2e-4`
- `gan_lr_d = 1e-4`
- `fid_every = 10`

### 2. Generate a Synthetic Pool

After GAN training, the best checkpoint is used to generate a balanced synthetic dataset:

- `data_synth/apple/*.png`
- `data_synth/banana/*.png`
- `data_synth/orange/*.png`

### 3. Train the Classifier

The classifier is trained on:

- real-only data without classical augmentation (`real`)
- synthetic-only data without classical augmentation (`synth`)
- combined real + synthetic data without classical augmentation (`both`)
- optional real-only classical augmentation baseline (`real_aug`)

This is repeated across multiple training sizes to measure when synthetic data help and when they do not.

## Setup

Python dependencies are listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- The code defaults to `device="mps"` on Apple Silicon and falls back to CPU when MPS is unavailable.
- `torch`, `torchvision`, `numpy`, `scipy`, `scikit-learn`, and `matplotlib` are required for the full pipeline.

## Notebook Entry Points

Use `notebooks/` as the front door for all interactive work:

- `notebooks/01_data_setup.ipynb`
- `notebooks/02_train_gan.ipynb`
- `notebooks/03_generate_synthetic_data.ipynb`
- `notebooks/04_classifier_experiments.ipynb`
- `notebooks/05_task1_pipeline_and_results.ipynb`

These notebooks now contain the training and experiment workflow directly. The remaining Python files are only shared configuration, model definitions, or report utilities with no notebook counterpart.

## Reproducing the Pipeline

### Recommended Notebook Workflow

This is the end-to-end workflow that addresses the missing Task 1 pieces:

- GANs are retrained per data size
- synthetic pools are generated from the matching data budget
- `real` / `synth` / `both` are evaluated without classical augmentation
- `real_aug` is available as the optional non-generative baseline
- reported pipeline cost includes GAN training and synthetic image generation

Run the notebooks in order:

1. `notebooks/01_data_setup.ipynb`
2. `notebooks/02_train_gan.ipynb`
3. `notebooks/03_generate_synthetic_data.ipynb`
4. `notebooks/04_classifier_experiments.ipynb`
5. `notebooks/05_task1_pipeline_and_results.ipynb`

This writes:

- `data_splits/task1/n*/train/...`
- `runs/task1/gan/n*/...`
- `runs/task1/synth/n*/...`
- `runs/task1/clf/all_results.json`
- `runs/task1/pipeline_summary.json`

### Report Utility Commands

```bash
python scripts/export_report_tables.py
python scripts/plot_fid_svg.py
```

These commands regenerate:

- `reports/tables/dataset_summary.csv`
- `reports/tables/gan_fid_by_epoch.csv`
- `reports/tables/clf_results.csv`
- `reports/figures/fid_vs_epoch.svg`
## Remaining Python Files

- `config.py` - shared hyperparameters and path defaults
- `models/gan.py` - generator and projection discriminator definitions
- `models/classifier.py` - classifier architecture definition
- `scripts/export_report_tables.py` - exports CSV tables for the written report
- `scripts/plot_fid_svg.py` - generates the SVG FID plot for the report

## Report

The repository includes a self-contained report package under `reports/`:

- [reports/report.md](reports/report.md) - full master's-level report
- `reports/figures/` - figures used in the report
- `reports/tables/` - exported CSV tables

If you want the short version, start with the report's:

- Abstract
- Executive Summary
- Results
- Discussion

## Limitations

- The study uses a **single random seed**.
- Only **three classes** are evaluated.
- The synthetic-only setting reveals a clear domain gap.
- FID is computed on a relatively small validation split and should be interpreted as a monitoring signal, not a final verdict.

## Future Directions

- Multi-seed experiments with confidence intervals
- Stronger GAN baselines such as StyleGAN2-ADA
- Per-class FID or KID
- Better synthetic-to-real domain alignment
- Comparisons against stronger classical augmentation baselines

## Citation

If you reference this repository, cite the report in `reports/report.md` and mention that the project evaluates synthetic data utility using both intrinsic generative metrics and downstream classification accuracy.
