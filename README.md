# GAN Lab

This repository implements **Task 1** end-to-end for fruit image generation and classification (`apple`, `banana`, `orange`):

1. Train a Conditional GAN from scratch (no transfer learning).
2. Generate synthetic class-conditional images.
3. Train/evaluate a classifier under multiple training scenarios.
4. Analyze accuracy/F1/time trade-offs across different data amounts.

## Current Project Status

Validated reference run is the **CPU balanced run**:
- Log: `logs/06_classifier_balanced_cpu.log`
- CSV: `runs_classifier/amount_vs_accuracy_time_balanced_cpu.csv`

Best observed overall configuration in this run:
- `Real+Synth (100%)`: **Accuracy 0.9928**, **Macro F1 0.9723**, **Train Time 595.38s**

Low-data highlight:
- At `10%` real data, `Real+Synth` outperforms `Real-only`:
  - Accuracy: `0.9759` vs `0.9635`
  - Macro F1: `0.9383` vs `0.9126`

Historical comparison availability:
- A documented v1 (smaller dataset) vs v2 (expanded dataset) comparison is included in:
  - `FINAL_REPORT_TASK1.md`
  - `RESULTS_NOTES.md`

## Repository Layout

- `train_gan.py`: Conditional GAN training
- `generate_synthetic.py`: Synthetic image generation from GAN checkpoint
- `train_classifier.py`: Classifier experiments (real/synth/mixed/aug + ratio/count analysis)
- `split_dataset.py`: Train/val/test split creation from raw data
- `utils.py`: Shared transforms/device helpers
- `data/`: raw/split/synthetic data folders
- `runs_gan/`: GAN checkpoints and sample grids
- `runs_classifier/`: classifier result CSV files
- `logs/`: run logs

## Documentation Index

- `RUNBOOK_TERMINAL.md`: Commands to run the full pipeline from scratch
- `RESULTS_NOTES.md`: Compact metrics and comparison notes
- `PRESENTATION_PREP.md`: Slide structure and talking points
- `FINAL_REPORT_TASK1.md`: Full structured report for the assignment
- `scripts/generate_report_figures.py`: Generates report-ready comparison charts
- `scripts/generate_count_trend_figures.py`: Generates count-based trend charts (`200/400/800/1600` style)
- `scripts/compute_fid.py`: Computes FID for GAN output (Task 1 training part)
- `scripts/run_task1_gan_scale.py`: Automates GAN scaling runs + FID collection by data amount
- `scripts/generate_task2_three_case_report.py`: Produces Task 2 three-case table/figure (only synth, only real, real+synth)
- `reports/figures/`: Output charts used by the final report

## Professor Alignment (Task 1 / Task 2)

- Task 1 trend analysis:
  - Use multiple data amounts (e.g. `200 400 800 1600`) instead of only two dataset sizes.
  - Classifier trend command:
    - `FORCE_DEVICE=cpu python train_classifier.py --counts 200 400 800 1600 --skip-aug ...`
  - GAN-side quality metric:
    - `python scripts/compute_fid.py ...` (FID for generator output).
- Task 2 comparison:
  - Compare exactly three classification cases:
    - `only synthetic`, `only real`, `real + synthetic`
  - Generate concise Task 2 artifact:
    - `python scripts/generate_task2_three_case_report.py ...`

## Reproducibility Notes

- Use the virtual environment in `.venv`.
- For stable classifier evaluation, use CPU explicitly:
  - `FORCE_DEVICE=cpu`
- `train_classifier.py` uses class balancing by default (weighted sampler + class-weighted loss).
- Disable balancing baseline with:
  - `--disable-balancing`

## Git Notes

- Generated data and run artifacts are ignored to keep commits clean.
- See `RUNBOOK_TERMINAL.md` section `Git Commit Hygiene` for one-time cleanup and normal commit flow.

## Quick Start

See `RUNBOOK_TERMINAL.md` for exact commands and artifact paths.
