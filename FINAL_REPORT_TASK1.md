# Final Report - Task 1 (GAN-Based Data Generation and Classification)

## Executive Summary

1. A Conditional GAN was trained from scratch (no transfer learning) and used to generate `1200` synthetic fruit images (`400` per class).
2. In the latest validated CPU balanced run, the best overall classifier setup is `Real+Synth (100%)`:
   - Accuracy: **0.9925**
   - Macro F1: **0.9731**
3. In low-data settings, synthetic data provides clear gains (especially at `10%` and count `200`).
4. Task 1 quantitative generation metric was completed:
   - Overall FID: **178.7821**
   - Per-class FID: apple `165.7848`, banana `269.0328`, orange `260.0041`
5. Task 2 required three-case comparison is complete:
   - only synthetic, only real, real + synthetic.

## 1) Objective

Goal:
- Implement a GAN-based image generator from scratch (no transfer learning).
- Compare classifier performance under real-only, synth-only, and real+synth training.
- Analyze data-amount trend with both ratio-based and count-based views.

Classes:
- `apple`, `banana`, `orange`

## 2) Dataset and Split

From `logs/01_split.log`:
- Train: apple `15845`, banana `2160`, orange `746` (total `18751`)
- Val: apple `3395`, banana `462`, orange `160`
- Test: apple `3397`, banana `464`, orange `161`

Important:
- Dataset is heavily imbalanced toward apple.

## 3) Pipeline Overview

1. Split raw data (`split_dataset.py`).
2. Train Conditional GAN (`train_gan.py`).
3. Generate class-conditional synthetic images (`generate_synthetic.py`).
4. Train classifier scenarios (`train_classifier.py`).
5. Produce report tables/figures and compare trends.

## 4) GAN Evaluation (Task 1 Training Part)

### 4.1 Training Behavior

From `logs/02_train_gan.log`:
- Epoch 1: `loss_d=0.5270`, `loss_g=3.5130`
- Epoch 100: `loss_d=0.0500`, `loss_g=5.1522`

Interpretation:
- Discriminator becomes stronger over training.
- Generator remains trainable but under increasing adversarial pressure.
- No hard collapse/divergence was observed in the log.

### 4.2 Synthetic Data Volume

From `logs/03_generate_synthetic.log`:
- Apple: `400`
- Banana: `400`
- Orange: `400`
- Total: `1200`

### 4.3 FID Results

From `reports/fid_task1.json`:
- Overall FID: `178.7821`
- Per-class FID:
  - Apple: `165.7848`
  - Banana: `269.0328`
  - Orange: `260.0041`

Interpretation:
- Quality is better for apple than minority classes, consistent with class imbalance.

## 5) Classification Results - Ratio-Based Trend

Source:
- `logs/06_classifier_balanced_cpu.log`
- `runs_classifier/amount_vs_accuracy_time_balanced_cpu.csv`

### 5.1 Summary Table

| Ratio | Scenario | Aug | Accuracy | Macro F1 | Train Time (s) |
|---|---|---|---:|---:|---:|
| fixed | synth_only | no | 0.9724 | 0.9269 | 24.54 |
| 10% | real_only | no | 0.9635 | 0.9126 | 52.42 |
| 10% | real_plus_synth | no | 0.9846 | 0.9549 | 69.82 |
| 10% | real_only | yes | 0.8667 | 0.7367 | 55.68 |
| 25% | real_only | no | 0.9811 | 0.9446 | 130.44 |
| 25% | real_plus_synth | no | 0.9863 | 0.9493 | 139.80 |
| 25% | real_only | yes | 0.9664 | 0.9218 | 137.97 |
| 50% | real_only | no | 0.9891 | 0.9654 | 263.07 |
| 50% | real_plus_synth | no | 0.9888 | 0.9630 | 266.89 |
| 50% | real_only | yes | 0.9861 | 0.9556 | 277.39 |
| 100% | real_only | no | 0.9858 | 0.9511 | 530.13 |
| 100% | real_plus_synth | no | 0.9925 | 0.9731 | 524.49 |
| 100% | real_only | yes | 0.9896 | 0.9609 | 559.77 |

### 5.2 Real+Synth vs Real-only (No Aug)

| Ratio | Delta Accuracy | Delta Macro F1 | Delta Time (s) |
|---|---:|---:|---:|
| 10% | +0.0211 | +0.0423 | +17.40 |
| 25% | +0.0052 | +0.0047 | +9.36 |
| 50% | -0.0003 | -0.0024 | +3.82 |
| 100% | +0.0067 | +0.0220 | -5.64 |

Interpretation:
- Strong gain in low-data regime (`10%`).
- Near-neutral effect around `50%`.
- At full data (`100%`), real+synth is both better and slightly faster in this run.

### 5.3 Figures

![Accuracy vs Ratio](reports/figures/accuracy_vs_ratio_cpu_balanced.png)

![Macro F1 vs Ratio](reports/figures/macrof1_vs_ratio_cpu_balanced.png)

![Training Time vs Ratio](reports/figures/time_vs_ratio_cpu_balanced.png)

## 6) Classification Results - Count-Based Trend (Professor Requirement)

Source:
- `logs/08_classifier_counts_cpu.log`
- `runs_classifier/task1_amount_trend_counts_cpu.csv`

### 6.1 Main Table (`200/400/800/1600`)

| Real Train Count | Scenario | Accuracy | Macro F1 | Train Time (s) |
|---:|---|---:|---:|---:|
| fixed | synth_only | 0.9724 | 0.9269 | 24.39 |
| 200 | real_only | 0.9030 | 0.7810 | 5.20 |
| 200 | real_plus_synth | 0.9630 | 0.9045 | 28.79 |
| 400 | real_only | 0.9446 | 0.8662 | 11.18 |
| 400 | real_plus_synth | 0.9615 | 0.8938 | 33.48 |
| 800 | real_only | 0.9324 | 0.8294 | 21.57 |
| 800 | real_plus_synth | 0.9779 | 0.9355 | 42.93 |
| 1600 | real_only | 0.9463 | 0.8721 | 44.11 |
| 1600 | real_plus_synth | 0.9818 | 0.9458 | 61.96 |

### 6.2 Delta Table (Real+Synth - Real-only)

| Count | Delta Accuracy | Delta Macro F1 | Delta Time (s) |
|---:|---:|---:|---:|
| 200 | +0.0600 | +0.1235 | +23.59 |
| 400 | +0.0169 | +0.0276 | +22.30 |
| 800 | +0.0455 | +0.1061 | +21.36 |
| 1600 | +0.0355 | +0.0737 | +17.85 |

Interpretation:
- Across all tested counts, `real_plus_synth` outperforms `real_only` on both Accuracy and Macro F1.
- This directly satisfies the professor's requirement to show trend over multiple dataset sizes.

### 6.3 Figures

![Accuracy vs Count](reports/figures/accuracy_vs_count.png)

![Macro F1 vs Count](reports/figures/macrof1_vs_count.png)

![Training Time vs Count](reports/figures/time_vs_count.png)

## 7) Task 2 Three-Case Comparison (Required Format)

Source:
- `reports/task2_three_case_comparison.md`

Selected real setting: `100%`

| Case | Accuracy | Macro F1 | Train Time (s) |
|---|---:|---:|---:|
| Only Synthetic | 0.9724 | 0.9269 | 24.54 |
| Only Real | 0.9858 | 0.9511 | 530.13 |
| Real + Synthetic | 0.9925 | 0.9731 | 524.49 |

Interpretation:
- All required Task 2 scenarios are covered exactly once.
- Real + synthetic gives best performance in this run.

## 8) Historical v1 vs v2 Comparison

This remains an informative cross-run comparison (not strict single-variable ablation).

### 8.1 Main 100% Scenarios

| Scenario | v1 Accuracy | v2 Accuracy | Delta | v1 Macro F1 | v2 Macro F1 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Synth-only | 0.9039 | 0.9724 | +0.0685 | 0.9045 | 0.9269 | +0.0224 |
| Real-only (100%) | 0.9476 | 0.9858 | +0.0382 | 0.9462 | 0.9511 | +0.0049 |
| Real+Synth (100%) | 0.8690 | 0.9925 | +0.1235 | 0.8650 | 0.9731 | +0.1081 |
| Real-only + Aug (100%) | 0.9301 | 0.9896 | +0.0595 | 0.9303 | 0.9609 | +0.0306 |

### 8.2 Ratio-Level Accuracy Delta (v2-v1)

| Ratio | Real-only Delta | Real+Synth Delta |
|---|---:|---:|
| 10% | +0.1731 | +0.1549 |
| 25% | +0.1296 | +0.0780 |
| 50% | +0.1070 | +0.0892 |
| 100% | +0.0382 | +0.1235 |

## 9) Conclusions

1. Conditional GAN from scratch produces class-informative synthetic data.
2. Synthetic data consistently improves low-data classification performance.
3. Count-based trend (`200/400/800/1600`) confirms synthetic benefit across all tested amounts.
4. FID indicates class-dependent generation quality; minority classes remain weaker.
5. For defense, emphasize Macro F1 and class imbalance impact, not Accuracy alone.

## 10) Reproducibility Commands

### 10.1 Ratio Run

```bash
FORCE_DEVICE=cpu python train_classifier.py \
  --epochs 20 \
  --ratios 0.1 0.25 0.5 1.0 \
  --num-workers 0 \
  --out-csv runs_classifier/amount_vs_accuracy_time_balanced_cpu.csv \
  | tee logs/06_classifier_balanced_cpu.log
```

### 10.2 Count Run

```bash
FORCE_DEVICE=cpu python train_classifier.py \
  --epochs 20 \
  --counts 200 400 800 1600 \
  --skip-aug \
  --num-workers 0 \
  --out-csv runs_classifier/task1_amount_trend_counts_cpu.csv \
  | tee logs/08_classifier_counts_cpu.log
```

### 10.3 FID Run

```bash
TORCH_HOME=/Users/ardaerdogan/Desktop/gan-lab/.torch-cache \
/Users/ardaerdogan/Desktop/gan-lab/.venv/bin/python \
  scripts/compute_fid.py \
  --real-dir data/split/train \
  --fake-dir data/synthetic \
  --weights-path /Users/ardaerdogan/Desktop/gan-lab/.torch-cache/hub/checkpoints/inception_v3_google-0cc3c7bd.pth \
  --per-class \
  --out-json reports/fid_task1.json
```
