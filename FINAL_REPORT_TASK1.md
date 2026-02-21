# Final Report - Task 1 (GAN-Based Data Generation and Classification)

## Executive Summary

1. A Conditional GAN was trained from scratch (no transfer learning) and used to generate 1200 synthetic fruit images (400 per class).
2. In the validated CPU balanced run, `Real+Synth (100%)` achieved the best overall result: **Accuracy 0.9928**, **Macro F1 0.9723**.
3. Synthetic data provided clear gains in low-data settings (`10%`, `25%`), while one mid-regime (`50%`) showed negative transfer.
4. Training with synthetic data increases computation time, but the overhead decreases at higher real-data ratios.
5. Historical v1 (smaller dataset) vs v2 (expanded dataset) comparison shows substantial improvements in most scenarios, especially `Real+Synth (100%)`.

## 1) Objective

Goal:
- Implement a GAN-based image generator from scratch (no transfer learning).
- Compare classifier performance when trained on real-only, synth-only, and real+synth data.
- Analyze how training data amount affects accuracy, macro F1, and computation time.

Classes:
- `apple`, `banana`, `orange`

## 2) Pipeline Overview

1. Build train/val/test split from raw data (`split_dataset.py`).
2. Train Conditional GAN (`train_gan.py`).
3. Generate synthetic dataset (`generate_synthetic.py`).
4. Train classifier under multiple scenarios (`train_classifier.py`).
5. Compare metrics across data ratios and augmentation strategies.

## 3) Model Implementations

### 3.1 GAN Generator/Discriminator

- Conditional GAN trained from scratch.
- Input image size: `32x32`
- No transfer learning used.
- Saved outputs:
  - Checkpoints: `runs_gan/ckpt_epoch_*.pt`
  - Samples: `runs_gan/samples_epoch_*.png`

### 3.2 Classifier

- CNN classifier trained under scenarios:
  - Real-only
  - Synth-only
  - Real+Synth
  - Real-only + classic augmentation (optional baseline)
- Evaluation metrics:
  - Accuracy
  - Macro F1
  - Confusion matrix
  - Training time (seconds)
- Class balancing:
  - Weighted sampler + class-weighted cross entropy (enabled by default)

## 4) Experiment Configuration

Reference run:
- Device: CPU (`FORCE_DEVICE=cpu`)
- Ratios: `10%`, `25%`, `50%`, `100%`
- Classifier epochs: `20`

Artifacts used:
- `logs/02_train_gan.log`
- `logs/03_generate_synthetic.log`
- `logs/06_classifier_balanced_cpu.log`
- `runs_classifier/amount_vs_accuracy_time_balanced_cpu.csv`

## 5) Evaluation of the GAN-Based Data Generator

The Conditional GAN was trained from scratch on the training split of the fruit dataset (apple, banana, orange). Transfer learning was not used.

### 5.1 Training Stability

GAN training losses:
- Epoch 1: `loss_d = 0.5270`, `loss_g = 3.5130`
- Epoch 100: `loss_d = 0.0504`, `loss_g = 5.3117`

Interpretation:
- Discriminator loss decreased overall.
- Generator loss increased overall with occasional spikes.
- No hard divergence was observed in the run.

### 5.2 Visual Quality Assessment

Visual inspection of generated samples (e.g., epochs 1, 50, 100) indicates:
- Clear class-level consistency (color and coarse shape)
- No obvious severe mode collapse
- Moderate high-frequency noise / minor artifacts

The generator captures coarse semantic structure (round red apples, elongated bananas, orange-colored oranges), while some texture-level imperfections remain.

### 5.3 Synthetic Dataset Generation

Generated synthetic samples:
- 400 synthetic apples
- 400 synthetic bananas
- 400 synthetic oranges

Total synthetic samples: **1200**

This dataset was used in downstream classifier experiments.

## 6) Classification Performance Comparison

Classifier scenarios evaluated:
- Real-only
- Synth-only
- Real+Synth
- Real-only + classic augmentation (optional)

### Summary Table (CPU Balanced Run)

| Ratio | Scenario | Aug | Accuracy | Macro F1 | Train Time (s) |
|---|---|---|---:|---:|---:|
| fixed | synth_only | no | 0.9597 | 0.8817 | 25.28 |
| 10% | real_only | no | 0.9635 | 0.9126 | 56.06 |
| 10% | real_plus_synth | no | 0.9759 | 0.9383 | 72.32 |
| 10% | real_only | yes | 0.8667 | 0.7367 | 57.58 |
| 25% | real_only | no | 0.9811 | 0.9446 | 129.49 |
| 25% | real_plus_synth | no | 0.9898 | 0.9639 | 143.32 |
| 25% | real_only | yes | 0.9664 | 0.9218 | 139.61 |
| 50% | real_only | no | 0.9891 | 0.9654 | 267.30 |
| 50% | real_plus_synth | no | 0.9821 | 0.9355 | 276.84 |
| 50% | real_only | yes | 0.9861 | 0.9556 | 284.67 |
| 100% | real_only | no | 0.9858 | 0.9511 | 590.53 |
| 100% | real_plus_synth | no | 0.9928 | 0.9723 | 595.38 |
| 100% | real_only | yes | 0.9896 | 0.9609 | 548.39 |

## 7) Results: Synth-Only Baseline

Synth-only achieved:
- Accuracy: **0.9597**
- Macro F1: **0.8817**
- Training time: **25.28 s**

Interpretation:
- Synthetic data contains meaningful class-discriminative information.
- Macro F1 below top mixed/real settings indicates distribution mismatch still exists.

## 8) Effect of Data Amount on Accuracy and Time

Ratios tested: `10%`, `25%`, `50%`, `100%`

### 8.0 Trend Figures

![Accuracy vs Ratio](reports/figures/accuracy_vs_ratio_cpu_balanced.png)

![Macro F1 vs Ratio](reports/figures/macrof1_vs_ratio_cpu_balanced.png)

![Training Time vs Ratio](reports/figures/time_vs_ratio_cpu_balanced.png)

### 8.1 Low-Data Regime (10%)

Real-only:
- Accuracy: 0.9635
- Macro F1: 0.9126
- Time: 56.06 s

Real+Synth:
- Accuracy: 0.9759
- Macro F1: 0.9383
- Time: 72.32 s

Change (Real+Synth - Real-only):
- +0.0124 accuracy
- +0.0257 macro F1
- +16.26 s training time

Interpretation:
- Synthetic data helps generalization when real data is scarce.

### 8.2 Moderate Data Regime (25%)

Real-only:
- Accuracy: 0.9811
- Macro F1: 0.9446
- Time: 129.49 s

Real+Synth:
- Accuracy: 0.9898
- Macro F1: 0.9639
- Time: 143.32 s

Change:
- +0.0087 accuracy
- +0.0193 macro F1
- +13.83 s time

Interpretation:
- Synthetic data still gives measurable gain.

### 8.3 Mid-High Data Regime (50%)

Real-only:
- Accuracy: 0.9891
- Macro F1: 0.9654
- Time: 267.30 s

Real+Synth:
- Accuracy: 0.9821
- Macro F1: 0.9355
- Time: 276.84 s

Change:
- -0.0070 accuracy
- -0.0299 macro F1
- +9.54 s time

Interpretation:
- At this ratio, synthetic data introduced harmful noise/artifacts for classifier learning.

### 8.4 Full Data Regime (100%)

Real-only:
- Accuracy: 0.9858
- Macro F1: 0.9511
- Time: 590.53 s

Real+Synth:
- Accuracy: 0.9928
- Macro F1: 0.9723
- Time: 595.38 s

Change:
- +0.0070 accuracy
- +0.0212 macro F1
- +4.85 s time

Interpretation:
- In this run, synthetic data again provided net benefit at full scale.

## 9) Comparison with Classical Augmentation

At 10% real data:
- Real-only + Classic Aug:
  - Accuracy: 0.8667
  - Macro F1: 0.7367
- GAN-based Real+Synth:
  - Accuracy: 0.9759
  - Macro F1: 0.9383

Interpretation:
- GAN-based augmentation clearly outperformed classic augmentation in low-data regime for this run.
- At higher ratios, classic augmentation became more competitive but generally did not surpass the best Real+Synth result.

## 10) Computational Cost Analysis

Synthetic mixing increased training time versus real-only:
- +16.26 s at 10%
- +13.83 s at 25%
- +9.54 s at 50%
- +4.85 s at 100%

Interpretation:
- There is a consistent compute overhead.
- In low-data regimes, performance gains justify this extra cost.

## 11) Final Conclusions

1. The Conditional GAN trained from scratch learned class-discriminative visual structure.
2. Synth-only classifier performance confirms generated data is meaningful.
3. GAN-based synthetic data is most consistently helpful in low-data settings.
4. Benefit is not monotonic across all ratios; distribution mismatch can hurt at some regimes.
5. Synthetic data is not a replacement for real data, but a practical supplement.

## 12) Historical Baseline vs Expanded Dataset (v1 vs v2)

To show dataset-scale impact, we compare:
- **v1 (historical baseline):** earlier smaller dataset run (documented in repository history).
- **v2 (current):** expanded dataset with archive integration, CPU balanced run.

Important:
- This is an informative cross-run comparison, not a strict controlled ablation.
- Setup differences (notably balancing/device/runtime scale) can affect outcomes.

### 12.0 Historical Comparison Figures

![v1 vs v2 Main Scenario Comparison](reports/figures/historical_v1_v2_main_100_comparison.png)

![v1 vs v2 Ratio Accuracy Comparison](reports/figures/historical_v1_v2_ratio_accuracy.png)

### 12.1 Main Scenario Comparison (100% setting)

| Scenario | v1 Accuracy | v2 Accuracy | Delta | v1 Macro F1 | v2 Macro F1 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Synth-only | 0.9039 | 0.9597 | +0.0558 | 0.9045 | 0.8817 | -0.0228 |
| Real-only (100%) | 0.9476 | 0.9858 | +0.0382 | 0.9462 | 0.9511 | +0.0049 |
| Real+Synth (100%) | 0.8690 | 0.9928 | +0.1238 | 0.8650 | 0.9723 | +0.1073 |
| Real-only + Aug (100%) | 0.9301 | 0.9896 | +0.0595 | 0.9303 | 0.9609 | +0.0306 |

### 12.2 Ratio-Level Accuracy Shift (Real-only vs Real+Synth)

| Ratio | Real-only Delta (v2-v1) | Real+Synth Delta (v2-v1) |
|---|---:|---:|
| 10% | +0.1731 | +0.1462 |
| 25% | +0.1296 | +0.0815 |
| 50% | +0.1070 | +0.0825 |
| 100% | +0.0382 | +0.1238 |

Interpretation:
- Accuracy is consistently higher in v2 across all ratios.
- The largest relative gain appears in `Real+Synth (100%)`.
- This supports the claim that expanded data and improved training configuration significantly strengthened classifier outcomes.

## 13) Practical Next Steps

1. Run an unbalanced baseline (`--disable-balancing`) for additional ablation.
2. Add quantitative generation metrics (e.g., FID/KID) if required by rubric.
3. Keep CPU-based logs as final reference to avoid device-dependent instability.

## 14) Figure Reproduction

To regenerate all report figures from current CSV results:

```bash
MPLCONFIGDIR=/Users/ardaerdogan/Desktop/gan-lab/.mplconfig \
  /Users/ardaerdogan/Desktop/gan-lab/.venv/bin/python \
  scripts/generate_report_figures.py
```
