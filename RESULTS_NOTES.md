# Results Notes (Latest Valid Run)

This file summarizes the latest reliable experiment outputs for Task 1.

## Run Reference

- GAN log: `logs/02_train_gan.log`
- Synthetic generation log: `logs/03_generate_synthetic.log`
- Classifier log (reference): `logs/06_classifier_balanced_cpu.log`
- Classifier table: `runs_classifier/amount_vs_accuracy_time_balanced_cpu.csv`

## 1) GAN Training and Synthetic Data

### 1.1 Training Behavior

- Epoch 1: `loss_d=0.5270`, `loss_g=3.5130`
- Epoch 100: `loss_d=0.0504`, `loss_g=5.3117`

Interpretation:
- Discriminator became stronger over time (`loss_d` decreased).
- Generator continued learning but with increasing adversarial pressure (`loss_g` increased).
- No hard divergence was observed in the run logs.

### 1.2 Generated Data Volume

From `logs/03_generate_synthetic.log`:
- Apple: `400`
- Banana: `400`
- Orange: `400`
- Total: `1200`

## 2) Classifier Scenario Results (CPU, Balanced)

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

## 3) Real+Synth vs Real-only (No Aug)

| Ratio | Delta Accuracy | Delta Macro F1 | Delta Time (s) |
|---|---:|---:|---:|
| 10% | +0.0124 | +0.0257 | +16.26 |
| 25% | +0.0087 | +0.0193 | +13.83 |
| 50% | -0.0070 | -0.0299 | +9.54 |
| 100% | +0.0070 | +0.0212 | +4.85 |

Interpretation:
- Synthetic data helps clearly at `10%` and `25%`.
- At `50%`, synthetic data hurts this run.
- At `100%`, synthetic data again improves both accuracy and macro F1.

## 4) Classic Augmentation vs Real-only (No Synth)

| Ratio | Delta Accuracy | Delta Macro F1 | Delta Time (s) |
|---|---:|---:|---:|
| 10% | -0.0968 | -0.1759 | +1.52 |
| 25% | -0.0147 | -0.0228 | +10.12 |
| 50% | -0.0030 | -0.0098 | +17.37 |
| 100% | +0.0038 | +0.0098 | -42.14 |

Interpretation:
- Classical augmentation is weak at low data in this run.
- At full data, it is competitive but still below `Real+Synth (100%)` on macro F1.

## 5) Quick Conclusions

1. GAN-generated data is useful and class-informative (`synth_only` remains high).
2. GAN synthetic data is most useful in low-data regimes (`10%`, `25%`).
3. Effects are not monotonic (`50%` regression exists).
4. Compute cost increases with synthetic mixing, but overhead shrinks at higher data ratios.

## 6) Historical Baseline Comparison (v1 Small vs v2 Expanded)

Baseline source:
- v1 metrics were documented in previous project notes (repository history).
- v2 metrics are from `runs_classifier/amount_vs_accuracy_time_balanced_cpu.csv`.

### 6.1 100% Scenario Comparison

| Scenario | v1 Acc | v2 Acc | Delta | v1 F1 | v2 F1 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Synth-only | 0.9039 | 0.9597 | +0.0558 | 0.9045 | 0.8817 | -0.0228 |
| Real-only | 0.9476 | 0.9858 | +0.0382 | 0.9462 | 0.9511 | +0.0049 |
| Real+Synth | 0.8690 | 0.9928 | +0.1238 | 0.8650 | 0.9723 | +0.1073 |
| Real+ClassicAug | 0.9301 | 0.9896 | +0.0595 | 0.9303 | 0.9609 | +0.0306 |

### 6.2 Ratio-Level Accuracy Delta (v2-v1)

| Ratio | Real-only Delta | Real+Synth Delta |
|---|---:|---:|
| 10% | +0.1731 | +0.1462 |
| 25% | +0.1296 | +0.0815 |
| 50% | +0.1070 | +0.0825 |
| 100% | +0.0382 | +0.1238 |

Interpretation:
- Expanded dataset + updated training setup produce consistently higher accuracies.
- This section should be presented as **cross-run comparison**, not strict one-variable ablation.
