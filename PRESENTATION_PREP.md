# Presentation Prep (Updated)

## 1) Slide Flow

## Slide 1 - Problem

Research question:
- Can GAN-generated synthetic data improve fruit classification versus using only real data?

Task constraints:
- Conditional GAN must be trained from scratch.
- Transfer learning is not allowed.

## Slide 2 - Pipeline

1. Build split from raw data (`split_dataset.py`).
2. Train Conditional GAN (`train_gan.py`).
3. Generate synthetic dataset (`generate_synthetic.py`).
4. Train classifier in multiple scenarios (`train_classifier.py`).
5. Compare accuracy, macro F1, confusion matrix, and training time.

## Slide 3 - GAN Setup

- Model: Conditional GAN
- Classes: `apple`, `banana`, `orange`
- Image size: `32x32`
- Epochs: `100`
- No transfer learning

## Slide 4 - GAN Evaluation

From `logs/02_train_gan.log`:
- Epoch 1: `loss_d=0.5270`, `loss_g=3.5130`
- Epoch 100: `loss_d=0.0504`, `loss_g=5.3117`

From `runs_gan/samples_epoch_*.png`:
- Good class-level consistency
- No clear severe mode collapse
- Some texture/artifact noise remains

From `logs/03_generate_synthetic.log`:
- Generated `1200` images total (`400` per class)

## Slide 5 - Classification Scenarios

- Synth-only (fixed baseline)
- Real-only (`10/25/50/100%`)
- Real+Synth (`10/25/50/100%`)
- Real-only + classic augmentation (optional baseline)

All reference metrics are from CPU balanced run:
- `logs/06_classifier_balanced_cpu.log`
- `runs_classifier/amount_vs_accuracy_time_balanced_cpu.csv`

## Slide 6 - Key Results

- Synth-only: `Acc 0.9597`, `F1 0.8817`
- Real-only (100%): `Acc 0.9858`, `F1 0.9511`
- Real+Synth (100%): `Acc 0.9928`, `F1 0.9723` (best overall)

Low-data gains of Real+Synth vs Real-only:
- `10%`: `+0.0124 Acc`, `+0.0257 F1`
- `25%`: `+0.0087 Acc`, `+0.0193 F1`

Counterexample:
- `50%`: Real+Synth underperforms (`-0.0070 Acc`, `-0.0299 F1`)

## Slide 7 - Time vs Performance

Real+Synth increases training time:
- `+16.26s` at `10%`
- `+13.83s` at `25%`
- `+9.54s` at `50%`
- `+4.85s` at `100%`

Message:
- Synthetic data adds cost; benefit depends on data regime.

## Slide 8 - Conclusions

1. Conditional GAN learned useful class-discriminative features.
2. Synthetic-only classifier is strong but still below best mixed/real scenarios.
3. Synthetic data is most helpful in low-data regimes.
4. Synthetic data is a supplement, not a replacement for real data.

## Slide 9 - Historical Baseline vs Expanded Data (Optional but Strong)

Use this if jury asks, \"Did more data help?\".

v1 (small dataset, historical baseline) vs v2 (expanded dataset, current run):
- Real-only (100%): `0.9476 -> 0.9858` (`+0.0382`)
- Real+Synth (100%): `0.8690 -> 0.9928` (`+0.1238`)
- 10% Real-only: `0.7904 -> 0.9635` (`+0.1731`)
- 10% Real+Synth: `0.8297 -> 0.9759` (`+0.1462`)

Important line for scientific integrity:
- \"This is a cross-run historical comparison; not a strict controlled single-variable ablation.\"

## 2) Likely Questions and Ready Answers

Q: Why no transfer learning?
A: Task requirement explicitly forbids it.

Q: Why use macro F1, not only accuracy?
A: Macro F1 is more robust for class imbalance and per-class fairness.

Q: Is GAN always helpful?
A: No. It helps in some regimes (`10%`, `25%`, `100%`) and can hurt in others (`50%`).

Q: Main practical takeaway?
A: Use GAN data when real data is limited; validate per regime due potential distribution mismatch.
