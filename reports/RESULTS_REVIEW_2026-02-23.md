# Results Review Report (2026-02-23)

## Scope

Reviewed artifacts:
- `logs/01_split.log`
- `logs/02_train_gan.log`
- `logs/03_generate_synthetic.log`
- `logs/06_classifier_balanced_cpu.log`
- `logs/08_classifier_counts_cpu.log`
- `runs_classifier/amount_vs_accuracy_time_balanced_cpu.csv`
- `runs_classifier/task1_amount_trend_counts_cpu.csv`
- `reports/fid_task1.json`
- `reports/task2_three_case_comparison.md`

## Requirement Status

1. Task 1 trend over multiple dataset sizes: **Done**
   - Ratio view: `10/25/50/100%`
   - Count view: `200/400/800/1600`
2. Task 1 FID metric on GAN output: **Done**
   - Overall and per-class FID computed.
3. Task 2 three-case comparison only: **Done**
   - only synthetic / only real / real + synthetic.
4. Reportable artifacts + scripts for defense: **Done**
   - Logs, CSVs, figures, and scripts are present.

## Key Quantitative Findings

### Ratio-based classifier (CPU, balanced)

- Best overall (`100%`, real + synthetic):
  - Accuracy: `0.9925`
  - Macro F1: `0.9731`
- Low-data gain (`10%`, real + synthetic vs real-only):
  - `+0.0211` Accuracy
  - `+0.0423` Macro F1
- Mid regime (`50%`) is near-neutral:
  - `-0.0003` Accuracy
  - `-0.0024` Macro F1

### Count-based trend (`200/400/800/1600`)

For all four counts, `real + synthetic` beats `real-only` on both Accuracy and Macro F1.

Largest gains:
- Count `200`: `+0.0600` Accuracy, `+0.1235` Macro F1.
- Count `800`: `+0.0455` Accuracy, `+0.1061` Macro F1.

### FID (GAN quality)

From `reports/fid_task1.json`:
- Overall FID: `178.7821`
- Per-class:
  - Apple: `165.7848`
  - Banana: `269.0328`
  - Orange: `260.0041`

Interpretation:
- Synthetic quality is weaker on minority classes (`banana`, `orange`) than on `apple`.

## Technical Comments

1. Evidence now supports the thesis that synthetic data helps most in low-data settings.
2. Data imbalance is the dominant risk:
   - Train split (`apple` heavy) and per-class FID both point to class-dependent generation quality.
3. Macro F1 must be emphasized alongside Accuracy in the defense.
4. Acknowledge that FID is computed against an imbalanced real distribution; this can bias class-level outcomes.

## Defense Recommendations

1. Lead with count-based learning curve (`200/400/800/1600`) because this directly matches professor wording.
2. Show Task 2 three-case table early (`reports/task2_three_case_comparison.md`) to demonstrate strict compliance.
3. When asked about quality metrics, use both:
   - classifier uplift (downstream utility),
   - FID/per-class FID (generation fidelity).
