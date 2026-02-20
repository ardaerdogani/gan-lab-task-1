# Results Notes

Bu dosya, rapor/sunum sırasında hızlı referans için sonuç özetidir.

## 1) Ana Senaryolar (Notlanan Tam Koşu)

| Scenario | Accuracy | Macro F1 | Train Time (s) | Yorum |
|---|---:|---:|---:|---|
| Real-only | 0.9432 | 0.9432 | 56.72 | En güçlü referans |
| Synth-only | 0.8777 | 0.8775 | 23.88 | Kullanılabilir ama real'den düşük |
| Real+Synth | 0.9432 | 0.9409 | 81.88 | Real-only ile aynı accuracy, daha yüksek maliyet |

Yorum:
- Synthetic-only sonuçları, generator'ın class-discriminative örüntüleri öğrendiğini gösterir.
- Real-only ve Real+Synth aynı accuracy verdi; yeterli real veri varken synthetic faydası sınırlı kalabilir.
- Synth-only ile real-only arasındaki performans farkı, distribution mismatch ile tutarlıdır.

## 2) Low-Data Smoke Sonucu

Kaynak: `runs_classifier/smoke.csv`

| Ratio | Scenario | Accuracy | Macro F1 | Train Time (s) |
|---|---|---:|---:|---:|
| fixed | synth_only | 0.4105 | 0.3654 | 1.34 |
| 10% | real_only | 0.3668 | 0.1789 | 0.32 |
| 10% | real_plus_synth | 0.5371 | 0.5166 | 1.47 |

Yorum:
- `%10 real` koşulunda `real_plus_synth`, `real_only`’den bariz yüksek.
- Bu, "synthetic data low-data regime’de daha faydalı" sonucunu destekler.

## 3) Sunumda Kullanılacak Kısa Cümleler

- "Synthetic data is not a replacement for real data."
- "Its benefit becomes more visible in low-data regimes."
- "When real data is sufficient, synthetic data may increase cost without clear accuracy gain."
- "The generator captures coarse class-level structure, but high-frequency details remain noisy."

## 4) Ek Metod Notları

- Preprocessing, GAN ve classifier arasında tutarlılık için `utils.py` üzerinden merkezi hale getirildi.
- Dataset hafif dengesizdir, ancak şiddetli değildir.
- Synthetic veri class-balanced üretilerek dengelemenin genelleme etkisi analiz edilmiştir.

## 5) Tam Analiz Çalıştırma Komutu

```bash
python train_classifier.py --epochs 20 --ratios 0.1 0.25 0.5 1.0 --out-csv runs_classifier/amount_vs_accuracy_time.csv
```

Opsiyonel augmentationsız:

```bash
python train_classifier.py --skip-aug
```

Tam koşu sonrası bu dosyayı `amount_vs_accuracy_time.csv` ile güncelle.
