# Defense Oral Q&A (1 Page)

Use this as a fast speaking sheet during code questions. Keep answers to 20-30 seconds each.

## 1) "Why did you not use transfer learning?"
Short answer:
- Task constraint: GAN must be trained from scratch.
- The Generator and Discriminator are custom and trained end-to-end from random init.
Point to code:
- `train_gan.py`
- `models_gan.py`

## 2) "How is this conditional GAN actually conditioned on class?"
Short answer:
- Training uses class labels in both Generator and Discriminator forward passes.
- During generation, we explicitly pass class IDs (`y`) per class.
Point to code:
- `train_gan.py` (label usage in training loop)
- `generate_synthetic.py` (`y = torch.full(...)`)

## 3) "How did you generate synthetic images in a reproducible way?"
Short answer:
- We load a fixed checkpoint and class mapping from checkpoint metadata.
- We generate fixed count per class and save deterministic-named PNGs.
Point to code:
- `generate_synthetic.py` (`load_generator`, `class_to_idx`, per-class loop)

## 4) "How did you create 200/400/800/1600 experiments fairly?"
Short answer:
- We use stratified subsetting by class proportions, not random global slicing.
- Allocation logic keeps class presence and fills remainder deterministically.
Point to code:
- `train_classifier.py` (`_allocate_per_class`, `make_stratified_subset_by_count`)
- `train_gan.py` (same count-based subsetting support)

## 5) "How did you handle class imbalance in classifier training?"
Short answer:
- Two mechanisms: weighted sampling and class-weighted loss.
- This reduces majority-class dominance and supports macro-F1 improvement.
Point to code:
- `train_classifier.py` (`build_balancing_weights`, `WeightedRandomSampler`, `CrossEntropyLoss(weight=...)`)

## 6) "Why report Macro F1, not only accuracy?"
Short answer:
- Dataset is heavily imbalanced toward apple.
- Accuracy can hide minority-class errors; macro F1 gives per-class-balanced performance.
Point to evidence:
- `logs/01_split.log` (class counts)
- `runs_classifier/amount_vs_accuracy_time_balanced_cpu.csv` (accuracy + macro_f1 side by side)

## 7) "Did synthetic data always help?"
Short answer:
- No. Benefit is regime-dependent.
- Strong gain at low data (10%), near-neutral around 50%, best overall at 100% mixed in this run.
Canonical numbers (CPU balanced run):
- `10%` real+synth vs real-only: `+0.0211` accuracy, `+0.0423` macro F1
- `100%` real+synth: `Acc 0.9925`, `Macro F1 0.9731`
Point to evidence:
- `runs_classifier/amount_vs_accuracy_time_balanced_cpu.csv`

## 8) "How did you evaluate generation quality?"
Short answer:
- We compute FID using Inception features and Frechet distance on feature statistics.
- We report both overall and per-class FID.
Canonical numbers:
- Overall FID: `178.7821`
- Per-class: apple `165.7848`, banana `269.0328`, orange `260.0041`
Point to code/evidence:
- `scripts/compute_fid.py`
- `reports/fid_task1.json`

## 9) "How did you satisfy Task 2 exactly as requested?"
Short answer:
- We generate a dedicated three-case artifact only:
  - only synthetic
  - only real
  - real + synthetic
- No extra case is mixed into the Task 2 summary.
Point to code/evidence:
- `scripts/generate_task2_three_case_report.py`
- `reports/task2_three_case_comparison.md`

## 10) "How can we reproduce your results quickly during defense?"
Short answer:
- Pipeline is script-based and logged end-to-end.
- For classifier reproducibility, use CPU mode and fixed seeds.
Point to runbook/evidence:
- `RUNBOOK_TERMINAL.md`
- `RESULTS_NOTES.md`
- `DEFENSE_SCRIPT_PACK.md`

## High-Integrity Closing Line (Use When Challenged)

"The core claim is practical, not absolute: synthetic data is most useful as a supplement in low-data settings, and we validate this with both classifier outcomes and FID-side quality checks."
