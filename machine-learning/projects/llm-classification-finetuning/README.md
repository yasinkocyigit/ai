# Proje Basligi

Bu dosya, projenin amaci ve isleyisi hakkinda temel bilgiler sunar. Yeni projeye baslarken bu sablon uzerinden ilerleyiniz.

## Proje Akis Semasi ve Adimlar

1. Veri Hazirligi (Data Phase):
   - Ham verileri data/ klasorune yukleyin.
   ```python
   # notebooks/ icinde veri okuma
   import pandas as pd
   train = pd.read_csv('../data/raw/train.csv')
   ```

2. Kesifsel Veri Analizi (EDA Phase):
   - notebooks/01-eda.ipynb dosyasinda veriyi inceleyin.
   - Eksik verileri doldurma ve ozellik muhendisligi kararlarini burada alin.

3. Otomasyon ve Scripting (Source Phase):
   - Kararlastirilan temizleme adimlarini src/preprocessing.py icine fonksiyon olarak yazin.
   ```python
   # src/preprocessing.py ornek yapisi:
   def clean_data(df):
       # Islemler...
       return df
   ```

4. Modelleme (Modelling Phase):
   - notebooks/02-modelling.ipynb dosyasinda src icindeki fonksiyonu cagirarak temiz veriyi elde edin ve modeli egitin.
   ```python
   # notebooks/02-modelling.ipynb icinde src'den fonksiyon cagirma
   import sys
   sys.path.append('../')
   from src.preprocessing import clean_data
   
   df_cleaned = clean_data(pd.read_csv('../data/raw/train.csv'))
   ```

5. Tahmin ve Submission (Delivery Phase):
   - Model sonuclarini submissions/ klasorune kaydedin.

## Klasör Yonergeleri

- data/raw/: Orjinal, hic el degmemis ham veriler.
- data/processed/: Temizlenmis ve modele hazir hale getirilmis veriler.
- notebooks/: Analiz ve deneme raporlari.
- src/: Tekrar kullanilabilir yardımcı kodlar.
- submissions/: Yarismaya gonderilen cikti dosyalari.

---
Not: Proje boyunca dökümantasyonun güncel tutulması tavsiye edilir.
