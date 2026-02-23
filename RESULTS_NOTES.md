# Results Notes (Updated: 2026-02-23)

This file summarizes the latest completed runs and the exact commands used.

## 1) Run Artifacts

- Split log: `logs/01_split.log`
- GAN training log: `logs/02_train_gan.log`
- Synthetic generation log: `logs/03_generate_synthetic.log`
- Classifier ratio log: `logs/06_classifier_balanced_cpu.log`
- Classifier count log: `logs/08_classifier_counts_cpu.log`
- Ratio CSV: `runs_classifier/amount_vs_accuracy_time_balanced_cpu.csv`
- Count CSV: `runs_classifier/task1_amount_trend_counts_cpu.csv`
- FID JSON: `reports/fid_task1.json`
- Task 2 three-case summary: `reports/task2_three_case_comparison.md`

## 2) Terminal Commands Executed (Current Notes)

### 2.0 End-to-End Command Sequence

```bash
cd /Users/ardaerdogan/Desktop/gan-lab
source /Users/ardaerdogan/Desktop/gan-lab/.venv/bin/activate

python split_dataset.py | tee logs/01_split.log
python train_gan.py --epochs 100 --num-workers 0 | tee logs/02_train_gan.log

python generate_synthetic.py \
  --ckpt-path runs_gan/ckpt_epoch_100.pt \
  --out-root data/synthetic \
  --num-per-class 400 \
  --batch-size 64 \
  | tee logs/03_generate_synthetic.log

FORCE_DEVICE=cpu python train_classifier.py \
  --epochs 20 \
  --ratios 0.1 0.25 0.5 1.0 \
  --num-workers 0 \
  --out-csv runs_classifier/amount_vs_accuracy_time_balanced_cpu.csv \
  | tee logs/06_classifier_balanced_cpu.log

FORCE_DEVICE=cpu python train_classifier.py \
  --epochs 20 \
  --counts 200 400 800 1600 \
  --skip-aug \
  --num-workers 0 \
  --out-csv runs_classifier/task1_amount_trend_counts_cpu.csv \
  | tee logs/08_classifier_counts_cpu.log

MPLCONFIGDIR=/Users/ardaerdogan/Desktop/gan-lab/.mplconfig \
  /Users/ardaerdogan/Desktop/gan-lab/.venv/bin/python \
  scripts/generate_report_figures.py

MPLCONFIGDIR=/Users/ardaerdogan/Desktop/gan-lab/.mplconfig \
  /Users/ardaerdogan/Desktop/gan-lab/.venv/bin/python \
  scripts/generate_count_trend_figures.py \
  --csv-path runs_classifier/task1_amount_trend_counts_cpu.csv \
  --out-dir reports/figures

MPLCONFIGDIR=/Users/ardaerdogan/Desktop/gan-lab/.mplconfig \
  /Users/ardaerdogan/Desktop/gan-lab/.venv/bin/python \
  scripts/generate_task2_three_case_report.py \
  --csv-path runs_classifier/amount_vs_accuracy_time_balanced_cpu.csv \
  --ratio '100%'

TORCH_HOME=/Users/ardaerdogan/Desktop/gan-lab/.torch-cache \
  /Users/ardaerdogan/Desktop/gan-lab/.venv/bin/python \
  scripts/compute_fid.py \
  --real-dir data/split/train \
  --fake-dir data/synthetic \
  --weights-path /Users/ardaerdogan/Desktop/gan-lab/.torch-cache/hub/checkpoints/inception_v3_google-0cc3c7bd.pth \
  --per-class \
  --out-json reports/fid_task1.json
```

### 2.1 Count Trend Figure Generation

```bash
MPLCONFIGDIR=/Users/ardaerdogan/Desktop/gan-lab/.mplconfig \
  /Users/ardaerdogan/Desktop/gan-lab/.venv/bin/python \
  scripts/generate_count_trend_figures.py \
  --csv-path runs_classifier/task1_amount_trend_counts_cpu.csv \
  --out-dir reports/figures
```

### 2.2 FID Calculation

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

## 3) Dataset Snapshot

From `logs/01_split.log`:
- Train: apple `15845`, banana `2160`, orange `746` (total `18751`)
- Val: apple `3395`, banana `462`, orange `160`
- Test: apple `3397`, banana `464`, orange `161`

Note:
- The dataset is strongly imbalanced toward `apple`.

## 4) GAN + Synthetic Data

From `logs/02_train_gan.log`:
- Epoch 1: `loss_d=0.5270`, `loss_g=3.5130`
- Epoch 100: `loss_d=0.0500`, `loss_g=5.1522`

From `logs/03_generate_synthetic.log`:
- Generated `1200` synthetic images (`400` per class).

Interpretation:
- Training remains stable (no hard collapse), but adversarial pressure grows over epochs.

## 5) FID Results (Task 1 Training Part)

From `reports/fid_task1.json`:
- Overall FID: `178.7821`
- Per-class FID:
  - Apple: `165.7848`
  - Banana: `269.0328`
  - Orange: `260.0041`

Interpretation:
- Generation quality is better for the majority class (`apple`) than minority classes.
- This aligns with class imbalance in real training data.

## 6) Classifier Results - Ratio Runs (CPU, Balanced)

Source: `runs_classifier/amount_vs_accuracy_time_balanced_cpu.csv`

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

### 6.1 Real+Synth vs Real-only (No Aug)

| Ratio | Delta Accuracy | Delta Macro F1 | Delta Time (s) |
|---|---:|---:|---:|
| 10% | +0.0211 | +0.0423 | +17.40 |
| 25% | +0.0052 | +0.0047 | +9.36 |
| 50% | -0.0003 | -0.0024 | +3.82 |
| 100% | +0.0067 | +0.0220 | -5.64 |

Interpretation:
- Strong low-data gain at `10%`.
- Near-neutral at `50%`.
- Best overall at `100%` with `real_plus_synth`.

## 7) Classifier Results - Count Trend (Professor Request)

Source: `runs_classifier/task1_amount_trend_counts_cpu.csv`

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

### 7.1 Real+Synth vs Real-only by Count

| Count | Delta Accuracy | Delta Macro F1 | Delta Time (s) |
|---:|---:|---:|---:|
| 200 | +0.0600 | +0.1235 | +23.59 |
| 400 | +0.0169 | +0.0276 | +22.30 |
| 800 | +0.0455 | +0.1061 | +21.36 |
| 1600 | +0.0355 | +0.0737 | +17.85 |

Interpretation:
- In all tested counts, synthetic mixing improves both accuracy and macro F1.
- This directly satisfies the professor’s multi-size trend requirement.

## 8) Task 2 Required Three-Case Comparison

From `reports/task2_three_case_comparison.md` (`100%` setting):
- Only Synthetic: `Acc 0.9724`, `F1 0.9269`
- Only Real: `Acc 0.9858`, `F1 0.9511`
- Real + Synthetic: `Acc 0.9925`, `F1 0.9731`

Interpretation:
- The required three-case comparison is complete and consistent.

## 9) Final Comments

1. Task 1 is now complete in the required format:
   - ratio trend,
   - count trend (`200/400/800/1600`),
   - FID included.
2. Task 2 requirement is met exactly with three scenarios.
3. Main risk is class imbalance; per-class FID indicates weaker quality for minority classes (`banana`, `orange`).
4. For defense, emphasize:
   - low-data gains from synthetic data,
   - FID evidence with class-wise gap,
   - why macro F1 is necessary under imbalance.
