# Professor Requirement Alignment Checklist

This checklist tracks what was requested and where the corresponding project artifact lives.

## 1) Task 1 Trend Must Use Multiple Data Amounts

Requirement:
- Do not compare only two dataset sizes.
- Show trend across several sizes (example: `200, 400, 800, 1600`).

Project support added:
- Count-based classifier trend mode:
  - `train_classifier.py --counts 200 400 800 1600 --skip-aug`
- Count-based trend figures:
  - `scripts/generate_count_trend_figures.py`

Expected outputs:
- `runs_classifier/task1_amount_trend_counts_cpu.csv`
- `reports/figures/accuracy_vs_count.png`
- `reports/figures/macrof1_vs_count.png`
- `reports/figures/time_vs_count.png`

## 2) Task 2 Must Compare Exactly Three Cases

Requirement:
- only synthetic
- only real
- real + synthetic

Project support added:
- Direct extraction/report script:
  - `scripts/generate_task2_three_case_report.py`

Expected outputs:
- `reports/task2_three_case_comparison.md`
- `reports/figures/task2_three_case_comparison.png`

## 3) FID Can Be Included for Task 1 Training Part

Requirement:
- FID for GAN/generation side (not classification model).

Project support added:
- FID computation script:
  - `scripts/compute_fid.py`
- Optional full automation across counts:
  - `scripts/run_task1_gan_scale.py`

Expected outputs:
- `reports/fid_task1.json` (single run)
- `runs_gan/task1_fid_by_count.csv` (multi-count trend)

## 4) Results Must Be in Report; Scripts Ready for Defense

Requirement:
- Provide report-form results.
- Keep scripts ready for defense code questions.

Project support added:
- Updated usage docs:
  - `README.md`
  - `RUNBOOK_TERMINAL.md`
- Dedicated report-ready Task 2 artifact generator:
  - `scripts/generate_task2_three_case_report.py`
