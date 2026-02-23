# Defense Script Pack

Use this during the defense to answer code-level questions quickly.

## 1) Keep These Scripts Open

Core pipeline:
- `split_dataset.py`
- `train_gan.py`
- `generate_synthetic.py`
- `train_classifier.py`

Evaluation and report generation:
- `scripts/compute_fid.py`
- `scripts/generate_task2_three_case_report.py`
- `scripts/generate_count_trend_figures.py`
- `scripts/generate_report_figures.py`

Model definitions:
- `models_gan.py`
- `data.py`
- `utils.py`

## 2) One-Line Responsibility Map

- `split_dataset.py`: builds train/val/test folder splits from raw fruit images.
- `train_gan.py`: trains conditional GAN from scratch (supports `--subset-count` for scale experiments).
- `generate_synthetic.py`: loads GAN checkpoint and writes class-conditional synthetic images.
- `train_classifier.py`: runs synth-only / real-only / real+synth experiments across ratios or counts.
- `scripts/compute_fid.py`: computes overall and per-class FID between real and synthetic images.
- `scripts/generate_task2_three_case_report.py`: extracts exactly three required Task 2 cases.
- `scripts/generate_count_trend_figures.py`: creates count-trend plots for `200/400/800/1600`.
- `scripts/generate_report_figures.py`: creates ratio-based report plots.

## 3) Most Likely Code Questions and Where to Point

Q: "How did you prevent transfer learning usage in GAN training?"
- Show `train_gan.py` and `models_gan.py` where Generator/Discriminator are defined and trained from scratch.
- We didn't use transfer learning; we defined the Generator and Discriminator ourselves, initialized them with random init, and optimized all parameters from scratch. There are no pretrained models or external checkpoint loading in the code.

Q: "How did you create smaller training-size experiments?"
- Show `make_stratified_subset_by_count` and `_allocate_per_class` in:
  - `train_gan.py` (GAN-side scaling)
  - `train_classifier.py` (classifier-side scaling)

Q: "How did you handle class imbalance?"
- Show `build_balancing_weights` in `train_classifier.py`:
  - weighted sampler (`WeightedRandomSampler`)
  - class-weighted `CrossEntropyLoss`

Q: "How did you ensure class-conditional generation?"
- Show `generate_synthetic.py`:
  - label tensor `y` per class
  - checkpoint `class_to_idx` usage

Q: "How did you compute FID?"
- Show `scripts/compute_fid.py`:
  - Inception feature extractor
  - Frechet distance on mean/covariance
  - optional `--per-class`

Q: "How did you enforce Task 2 exactly three cases?"
- Show `scripts/generate_task2_three_case_report.py`:
  - `synth_only`
  - `real_only`
  - `real_plus_synth`
  - selected ratio (typically `100%`)

## 4) Copy-Paste Commands for Live Defense

Run split:
```bash
python split_dataset.py | tee logs/01_split.log
```

Train GAN:
```bash
python train_gan.py --epochs 100 --num-workers 0 | tee logs/02_train_gan.log
```

Generate synthetic:
```bash
python generate_synthetic.py \
  --ckpt-path runs_gan/ckpt_epoch_100.pt \
  --out-root data/synthetic \
  --num-per-class 400 \
  --batch-size 64 \
  | tee logs/03_generate_synthetic.log
```

Classifier ratio run:
```bash
FORCE_DEVICE=cpu python train_classifier.py \
  --epochs 20 \
  --ratios 0.1 0.25 0.5 1.0 \
  --num-workers 0 \
  --out-csv runs_classifier/amount_vs_accuracy_time_balanced_cpu.csv \
  | tee logs/06_classifier_balanced_cpu.log
```

Classifier count run:
```bash
FORCE_DEVICE=cpu python train_classifier.py \
  --epochs 20 \
  --counts 200 400 800 1600 \
  --skip-aug \
  --num-workers 0 \
  --out-csv runs_classifier/task1_amount_trend_counts_cpu.csv \
  | tee logs/08_classifier_counts_cpu.log
```

FID:
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

Task 2 three-case artifact:
```bash
MPLCONFIGDIR=/Users/ardaerdogan/Desktop/gan-lab/.mplconfig \
  /Users/ardaerdogan/Desktop/gan-lab/.venv/bin/python \
  scripts/generate_task2_three_case_report.py \
  --csv-path runs_classifier/amount_vs_accuracy_time_balanced_cpu.csv \
  --ratio '100%'
```

Count trend figures:
```bash
MPLCONFIGDIR=/Users/ardaerdogan/Desktop/gan-lab/.mplconfig \
  /Users/ardaerdogan/Desktop/gan-lab/.venv/bin/python \
  scripts/generate_count_trend_figures.py \
  --csv-path runs_classifier/task1_amount_trend_counts_cpu.csv \
  --out-dir reports/figures
```

## 5) 60-Second Demo Strategy (If Asked to Show Code Fast)

1. Start with `train_classifier.py` (core comparison logic).
2. Jump to `generate_synthetic.py` (how synthetic data is produced by class).
3. Jump to `scripts/compute_fid.py` (quality metric side).
4. Close with `scripts/generate_task2_three_case_report.py` (requirement compliance).
