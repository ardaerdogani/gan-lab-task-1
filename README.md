# GAN Lab

Bu repo, meyve sınıfları (`apple`, `orange`, `banana`) için:
- veri split
- conditional GAN eğitimi
- sentetik veri üretimi
- classifier deneyleri (real/synth/mixed + low-data analizi)
akışını içerir.

Teknik notlar:
- Preprocessing adımları ortak `utils.py` içinde merkezileştirildi; GAN ve classifier tarafında tutarlı giriş dağılımı sağlandı.
- Dataset hafif dengesizdir, ancak dengesizlik şiddetli değildir.
- Synthetic üretim class-balanced yapıldı; hedef, dengelemenin classifier genellemesine etkisini ölçmektir.

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

Neden `32x32`?
- GAN eğitimini daha stabil hale getirir (mode collapse riski azalır).
- Düşük çözünürlükle deneyler daha hızlı tamamlanır.
- Aynı compute bütçesiyle daha fazla senaryo çalıştırıp `data size vs accuracy vs time` analizi yapılır.

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

## 5) Sonuçlar ve Yorum

### A) Notlanan ana sonuçlar (tam koşu)

| Scenario | Accuracy | Macro F1 | Train Time (s) |
|---|---:|---:|---:|
| Real-only (100%) | 0.9476 | 0.9462 | 13.24 |
| Synth-only (fixed) | 0.9039 | 0.9045 | 7.77 |
| Real+Synth (100%) | 0.8690 | 0.8650 | 17.61 |
| Real-only + Classic Aug (100%) | 0.9301 | 0.9303 | 12.71 |

Confusion Matrix (Real-only):

```text
[[76  1  7]
 [ 0 79  1]
 [ 3  0 62]]
```

Confusion Matrix (Synth-only):

```text
[[76  0  8]
 [ 8 69  3]
 [ 3  0 62]]
```

Confusion Matrix (Real+Synth):

```text
[[84  0  0]
 [ 9 70  1]
 [19  1 45]]
```

Kısa yorum:
- Synthetic-only modelinin `0.9039` accuracy elde etmesi, generator'ın sınıf-ayırt edici örüntüleri öğrendiğini gösterir.
- Bu koşuda `Real+Synth (100%)`, `Real-only (100%)`'den düşük çıkmıştır; synthetic eklemek her zaman kazanç getirmez.
- Real-only ile Synth-only arasındaki fark, real ve generated dağılımlar arasındaki mismatch ile tutarlıdır.

### B) Low-data ablation (tam koşu, `amount_vs_accuracy_time.csv`)

| Ratio | Real-only Acc | Real+Synth Acc | Real-only + Aug Acc | Real-only Time (s) | Real+Synth Time (s) |
|---|---:|---:|---:|---:|---:|
| 10% | 0.7904 | 0.8297 | 0.7336 | 4.12 | 9.38 |
| 25% | 0.8515 | 0.9083 | 0.7904 | 5.33 | 10.44 |
| 50% | 0.8821 | 0.8996 | 0.9301 | 7.05 | 11.56 |
| 100% | 0.9476 | 0.8690 | 0.9301 | 13.24 | 17.61 |

Kısa yorum:
- `10%` ve `25%` real veri koşullarında `Real+Synth`, `Real-only`'yi belirgin şekilde geçti.
- `50%` seviyesinde klasik augmentation (`0.9301`) en iyi sonucu verdi.
- `100%` seviyesinde en iyi sonuç `Real-only` (`0.9476`) oldu.

### C) `smoke` koşusu (hızlı doğrulama)

Kaynak: `runs_classifier/smoke.csv`

| Ratio | Scenario | Aug | Accuracy | Macro F1 | Train Time (s) |
|---|---|---|---|---|---|
| fixed | synth_only | no | 0.4105 | 0.3654 | 1.34 |
| 10% | real_only | no | 0.3668 | 0.1789 | 0.32 |
| 10% | real_plus_synth | no | 0.5371 | 0.5166 | 1.47 |

Kısa yorum:
- Low-data rejiminde (`%10 real`) `real_plus_synth`, `real_only`'den belirgin daha iyi.
- Synthetic veri özellikle veri az olduğunda yardımcı olabilir; bu bulgu task’in ana hipoteziyle uyumludur.

### D) GAN görsel kalite özeti

- Generator, renk ve genel şekil gibi coarse class-level özellikleri başarılı şekilde yakalar.
- Yüksek frekans detaylarında gürültü/artifact görülür (sınırlı veri boyutu + `32x32` etkisi).
- Çeşitlilik makul seviyededir; belirgin bir mode collapse gözlenmemiştir.

### E) Rapor için önerilen çıkarım

1. GAN verisi, real verinin yerine geçmez.
2. GAN verisi, real veri az olduğunda daha faydalıdır.
3. Synthetic eklemek eğitimi uzatır, bu yüzden accuracy-time dengesi raporlanmalıdır.
