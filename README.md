# GAN-Based Image Generation & Classification

Conditional Wasserstein GAN with Gradient Penalty (cWGAN-GP) for synthetic fruit image generation, plus a classification pipeline to compare real vs synthetic training data.

## Project Structure

```
gan-image-gen/
├── config.py                    # All hyperparameters (GAN + Classifier)
├── models/
│   ├── gan.py                   # Generator (cBN) + ProjectionCritic
│   └── classifier.py            # FruitCNN (from scratch, no transfer learning)
├── train_gan.py                 # WGAN-GP training loop + FID logging
├── train_classifier.py          # Classification: real / synth / both scenarios
├── scripts/
│   ├── generate_synth.py        # Generate synthetic dataset from trained G
│   ├── run_experiments.py       # Full experiment grid (5 sizes × 3 scenarios)
│   └── plot_results.py          # Accuracy/time plots + per-class F1 charts
├── data_final/
│   ├── train/                   # 1300 images/class
│   ├── val/                     # 159 images/class
│   └── test/                    # 492 images/class
└── data_synth/                  # Generated after Step 2
```

## Dataset

3 classes: **apple**, **banana**, **orange** (64×64 RGB)

| Split | Per Class | Total |
|-------|-----------|-------|
| Train | 1300      | 3900  |
| Val   | 159       | 477   |
| Test  | 492       | 1476  |

## Setup

```bash
conda create -n gan python=3.11 -y
conda activate gan
pip install torch torchvision scikit-learn matplotlib scipy
```

## Usage

### Step 1 — Train GAN

```bash
python train_gan.py
```

- cWGAN-GP with projection discriminator
- TTUR: G lr=1e-4, D lr=2e-4, Adam(β1=0.0, β2=0.9)
- n_critic=3, gradient penalty λ=10
- FID computed every 5 epochs on `data_final/val` (no augmentation)
- Checkpoints saved to `runs/gan/checkpoints/`
- Best FID checkpoint saved to `runs/gan/checkpoints/best_fid.pt`
- Sample grids saved to `runs/gan/`
- Training log: `runs/gan/train_log.json`

### Step 2 — Generate Synthetic Dataset

```bash
python scripts/generate_synth.py \
    --ckpt runs/gan/checkpoints/best_fid.pt \
    --n_per_class 1300 \
    --seed 42
```

Fixed seed ensures reproducibility. Run once, then freeze.

### Step 3 — Run Experiment Grid

```bash
python scripts/run_experiments.py
```

Runs 15 experiments (5 data sizes × 3 scenarios):

| Size/Class | Real | Synth | Real+Synth |
|------------|------|-------|------------|
| 100        | ✓    | ✓     | ✓          |
| 200        | ✓    | ✓     | ✓          |
| 400        | ✓    | ✓     | ✓          |
| 800        | ✓    | ✓     | ✓          |
| 1300       | ✓    | ✓     | ✓          |

Results saved to `runs/clf/all_results.json`.

### Step 4 — Generate Plots

```bash
python scripts/plot_results.py
```

Outputs to `runs/clf/plots/`:
- `accuracy_vs_size.png` — Test accuracy trend across data sizes
- `time_vs_size.png` — Training time comparison
- `per_class_f1.png` — Per-class F1 scores

## Architecture

### Generator (~1.5M params)
- Input: z (128-d) + class label
- 4× GenBlock: Upsample → Conv → ConditionalBatchNorm → ReLU
- Output: 3×64×64, Tanh

### Projection Critic (~4.9M params)
- 4× CriticBlock: Conv → LeakyReLU → Conv → LeakyReLU → AvgPool (with skip)
- Global sum pooling → linear + projection (Miyato & Koyama, 2018)

### Classifier (~813K params)
- 3× (Conv-BN-ReLU × 2 + Pool + Dropout) → AdaptiveAvgPool → FC head
- Trained from scratch, no pretrained weights

## Key Config (`config.py`)

| Parameter     | Value  | Description                    |
|---------------|--------|--------------------------------|
| `z_dim`       | 128    | Latent dimension               |
| `gan_epochs`  | 100    | GAN training epochs            |
| `n_critic`    | 3      | Critic updates per G update    |
| `gp_lambda`   | 10.0   | Gradient penalty coefficient   |
| `fid_every`   | 5      | FID evaluation interval         |
| `fid_eval_split` | val | FID real split (no augmentation) |
| `clf_epochs`  | 30     | Classifier training epochs     |
| `device`      | mps    | Apple Silicon GPU               |
