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
