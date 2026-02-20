# Results Notes

Bu dosya, rapor/sunum sırasında hızlı referans için sonuç özetidir.

## 1) Ana Senaryolar (Notlanan Tam Koşu)

| Scenario | Accuracy | Macro F1 | Train Time (s) | Yorum |
|---|---:|---:|---:|---|
| Real-only (100%) | 0.9476 | 0.9462 | 13.24 | En güçlü referans |
| Synth-only (fixed) | 0.9039 | 0.9045 | 7.77 | Kullanılabilir ama real'den düşük |
| Real+Synth (100%) | 0.8690 | 0.8650 | 17.61 | Bu koşuda düşüş gözlendi |
| Real-only + Classic Aug (100%) | 0.9301 | 0.9303 | 12.71 | Real-only'den düşük |

Yorum:
- Synthetic-only sonuçları, generator'ın class-discriminative örüntüleri öğrendiğini gösterir.
- Bu koşuda 100% real veri için synthetic eklemek performans düşürdü.
- Synth-only ile real-only arasındaki performans farkı, distribution mismatch ile tutarlıdır.

Confusion matrices (100% koşulları):

Real-only:
```text
[[76  1  7]
 [ 0 79  1]
 [ 3  0 62]]
```

Synth-only:
```text
[[76  0  8]
 [ 8 69  3]
 [ 3  0 62]]
```

Real+Synth:
```text
[[84  0  0]
 [ 9 70  1]
 [19  1 45]]
```

## 2) Low-Data Tam Koşu Sonucu (`amount_vs_accuracy_time.csv`)

| Ratio | Real-only Acc | Real+Synth Acc | Real-only + Aug Acc | Real-only Time (s) | Real+Synth Time (s) |
|---|---:|---:|---:|---:|---:|
| 10% | 0.7904 | 0.8297 | 0.7336 | 4.12 | 9.38 |
| 25% | 0.8515 | 0.9083 | 0.7904 | 5.33 | 10.44 |
| 50% | 0.8821 | 0.8996 | 0.9301 | 7.05 | 11.56 |
| 100% | 0.9476 | 0.8690 | 0.9301 | 13.24 | 17.61 |

Yorum:
- `10%` ve `25%` real veri koşullarında `Real+Synth` fayda sağladı.
- `50%` koşulunda klasik augmentation en iyi sonucu verdi.
- `100%` koşulunda en iyi model `Real-only` oldu.

## 3) Low-Data Smoke Sonucu

Kaynak: `runs_classifier/smoke.csv`

| Ratio | Scenario | Accuracy | Macro F1 | Train Time (s) |
|---|---|---:|---:|---:|
| fixed | synth_only | 0.4105 | 0.3654 | 1.34 |
| 10% | real_only | 0.3668 | 0.1789 | 0.32 |
| 10% | real_plus_synth | 0.5371 | 0.5166 | 1.47 |

Yorum:
- `%10 real` koşulunda `real_plus_synth`, `real_only`’den bariz yüksek.
- Bu, "synthetic data low-data regime’de daha faydalı" sonucunu destekler.

## 4) Sunumda Kullanılacak Kısa Cümleler

- "Synthetic data is not a replacement for real data."
- "Its benefit becomes more visible in low-data regimes."
- "When real data is sufficient, synthetic data may increase cost without clear accuracy gain."
- "The generator captures coarse class-level structure, but high-frequency details remain noisy."

## 5) Ek Metod Notları

- Preprocessing, GAN ve classifier arasında tutarlılık için `utils.py` üzerinden merkezi hale getirildi.
- Dataset hafif dengesizdir, ancak şiddetli değildir.
- Synthetic veri class-balanced üretilerek dengelemenin genelleme etkisi analiz edilmiştir.

## 6) Tam Analiz Çalıştırma Komutu

```bash
python train_classifier.py --epochs 20 --ratios 0.1 0.25 0.5 1.0 --out-csv runs_classifier/amount_vs_accuracy_time.csv
```

Opsiyonel augmentationsız:

```bash
python train_classifier.py --skip-aug
```

Tam koşu sonrası bu dosyayı `amount_vs_accuracy_time.csv` ile güncelle.
