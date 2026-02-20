# GAN Lab

Bu repo, meyve sınıfları (`apple`, `orange`, `banana`) için:
- veri split
- conditional GAN eğitimi
- sentetik veri üretimi
- classifier deneyleri (real/synth/mixed + low-data analizi)
akışını içerir.

## Ortam

Önerilen interpreter:

```bash
/Users/ardaerdogan/Desktop/gan-lab/.venv/bin/python
```

Aktifleştirme:

```bash
source /Users/ardaerdogan/Desktop/gan-lab/.venv/bin/activate
```

Not:
- Kod Apple Silicon için `mps` cihazını otomatik seçer (varsa).
- `train_classifier.py` için `--num-workers -1` varsayılanı otomatik worker seçimi yapar.

## 1) Dataset Split

Ham veri yapısı:

```text
data/raw/apples
data/raw/oranges
data/raw/bananas
```

Split üretimi:

```bash
python split_dataset.py
```

Çıkış:

```text
data/split/train/{apple,orange,banana}
data/split/val/{apple,orange,banana}
data/split/test/{apple,orange,banana}
```

## 2) GAN Eğitimi

```bash
python train_gan.py
```

Çıkışlar:
- checkpoint: `runs_gan/ckpt_epoch_XXX.pt`
- örnek görseller: `runs_gan/samples_epoch_XXX.png`

## 3) Sentetik Veri Üretimi

`generate_synthetic.py` içindeki `CKPT_PATH` ve `NUM_PER_CLASS` değerlerini gerekirse güncelle:
- `CKPT_PATH = runs_gan/ckpt_epoch_090.pt`
- `NUM_PER_CLASS = 400`

Çalıştır:

```bash
python generate_synthetic.py
```

Çıkış:

```text
data/synthetic/apple
data/synthetic/orange
data/synthetic/banana
```

## 4) Classifier Deneyleri

Tam deney (real/synth/mixed + low-data + opsiyonel classic aug):

```bash
python train_classifier.py --epochs 20 --ratios 0.1 0.25 0.5 1.0 --out-csv runs_classifier/amount_vs_accuracy_time.csv
```

Classic augmentation senaryosunu kapatmak için:

```bash
python train_classifier.py --skip-aug
```

Kısa smoke run:

```bash
python train_classifier.py --epochs 1 --ratios 0.1 --skip-aug --batch-size 128 --out-csv runs_classifier/smoke.csv
```

Çıktı tablo dosyası:
- `runs_classifier/amount_vs_accuracy_time.csv`

## Sunum Hazırlığı

Sunum akışı, olası mülakat soruları ve hazır cevaplar:
- `PRESENTATION_PREP.md`
