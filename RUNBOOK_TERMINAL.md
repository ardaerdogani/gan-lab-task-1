# Runbook (End-to-End, Reproducible)

## Setup

```bash
cd /Users/ardaerdogan/Desktop/gan-lab
source .venv/bin/activate
set -e
mkdir -p logs
```

## 0) Clean Start

```bash
rm -rf runs_gan runs_classifier
rm -rf data/split
mkdir -p data/split
find data/synthetic -type f -delete
```

## 1) Build Split

```bash
python split_dataset.py | tee logs/01_split.log
```

## 2) Train GAN

```bash
python train_gan.py | tee logs/02_train_gan.log
```

## 3) Validate Checkpoint for Synthetic Generation

```bash
ls -lh runs_gan/ckpt_epoch_090.pt
```

## 4) Generate Synthetic Dataset

```bash
python generate_synthetic.py | tee logs/03_generate_synthetic.log
```

## 5) Train Classifier (Balanced, CPU Reference)

```bash
FORCE_DEVICE=cpu python train_classifier.py \
  --epochs 20 \
  --ratios 0.1 0.25 0.5 1.0 \
  --num-workers 0 \
  --out-csv runs_classifier/amount_vs_accuracy_time_balanced_cpu.csv \
  | tee logs/06_classifier_balanced_cpu.log
```

## 5.1) Task 1 Trend by Data Amount (Professor Request)

Use absolute real-data amounts instead of only two dataset sizes:

```bash
FORCE_DEVICE=cpu python train_classifier.py \
  --epochs 20 \
  --counts 200 400 800 1600 \
  --skip-aug \
  --num-workers 0 \
  --out-csv runs_classifier/task1_amount_trend_counts_cpu.csv \
  | tee logs/08_classifier_counts_cpu.log
```

Generate count-based trend figures:

```bash
MPLCONFIGDIR=/Users/ardaerdogan/Desktop/gan-lab/.mplconfig \
  /Users/ardaerdogan/Desktop/gan-lab/.venv/bin/python \
  scripts/generate_count_trend_figures.py \
  --csv-path runs_classifier/task1_amount_trend_counts_cpu.csv
```

## 6) Optional Baseline (Unbalanced, CPU)

```bash
FORCE_DEVICE=cpu python train_classifier.py \
  --epochs 20 \
  --ratios 0.1 0.25 0.5 1.0 \
  --num-workers 0 \
  --disable-balancing \
  --out-csv runs_classifier/amount_vs_accuracy_time_unbalanced_cpu.csv \
  | tee logs/07_classifier_unbalanced_cpu.log
```

## 7) Files to Review and Share

- `logs/01_split.log`
- `logs/02_train_gan.log`
- `logs/03_generate_synthetic.log`
- `logs/06_classifier_balanced_cpu.log`
- `runs_classifier/amount_vs_accuracy_time_balanced_cpu.csv`
- `logs/08_classifier_counts_cpu.log` (Task 1 count-trend run)
- `runs_classifier/task1_amount_trend_counts_cpu.csv` (Task 1 count-trend table)
- `logs/07_classifier_unbalanced_cpu.log` (optional)
- `runs_classifier/amount_vs_accuracy_time_unbalanced_cpu.csv` (optional)

## 8) Generate Report Figures (Optional)

```bash
MPLCONFIGDIR=/Users/ardaerdogan/Desktop/gan-lab/.mplconfig \
  /Users/ardaerdogan/Desktop/gan-lab/.venv/bin/python \
  scripts/generate_report_figures.py
```

Generated files:
- `reports/figures/accuracy_vs_ratio_cpu_balanced.png`
- `reports/figures/macrof1_vs_ratio_cpu_balanced.png`
- `reports/figures/time_vs_ratio_cpu_balanced.png`
- `reports/figures/historical_v1_v2_main_100_comparison.png`
- `reports/figures/historical_v1_v2_ratio_accuracy.png`
- `reports/figures/accuracy_vs_count.png` (if count-trend CSV exists)
- `reports/figures/macrof1_vs_count.png` (if count-trend CSV exists)
- `reports/figures/time_vs_count.png` (if count-trend CSV exists)

## 8.1) Task 2 Required Three-Case Comparison

Professor requested only three Task 2 classification cases:
- only synthetic
- only real
- real + synthetic

Produce dedicated Task 2 markdown + chart from CSV:

```bash
/Users/ardaerdogan/Desktop/gan-lab/.venv/bin/python \
  scripts/generate_task2_three_case_report.py \
  --csv-path runs_classifier/amount_vs_accuracy_time_balanced_cpu.csv \
  --ratio 100%
```

Outputs:
- `reports/task2_three_case_comparison.md`
- `reports/figures/task2_three_case_comparison.png`

## 8.2) Task 1 GAN FID (Training Part)

`compute_fid.py` requires Inception v3 pretrained weights. If automatic download is unavailable, pass a local path.
Recommended (validated) command:

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

## 8.3) Full Task 1 GAN Scale Automation (Optional)

Runs GAN at multiple data amounts and collects FID trend:

```bash
/Users/ardaerdogan/Desktop/gan-lab/.venv/bin/python \
  scripts/run_task1_gan_scale.py \
  --counts 200 400 800 1600 \
  --epochs 30 \
  --weights-path /ABSOLUTE/PATH/inception_v3_google-0cc3c7bd.pth
```

Main output:
- `runs_gan/task1_fid_by_count.csv`

## Notes

- If you type lines like `# 2) ...` directly in zsh and get `parse error near ')'`, do not run section headers as commands.
- If needed, enable interactive comments once per shell:
  - `setopt interactive_comments`

## 9) Git Commit Hygiene (Prevents 10k+ File Commits)

One-time cleanup commit (recommended):
- Purpose: stop tracking generated data/artifacts so future commits remain small.

```bash
git add .gitignore
git add -A data/raw data/split data/synthetic runs_gan runs_classifier logs __pycache__ reports/figures .mplconfig
git add README.md RESULTS_NOTES.md PRESENTATION_PREP.md RUNBOOK_TERMINAL.md FINAL_REPORT_TASK1.md scripts/generate_report_figures.py
git commit -m "chore: stop tracking generated artifacts and update reporting docs"
```

After this cleanup commit, normal workflow:

```bash
git add README.md RESULTS_NOTES.md PRESENTATION_PREP.md FINAL_REPORT_TASK1.md
git commit -m "docs: update report and analysis"
```
