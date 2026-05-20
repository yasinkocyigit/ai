# Titanic - Machine Learning from Disaster

Bu proje, Kaggle platformunun efsanevi "Getting Started" yarismasinda yer alan Titanic yolcu verilerinin analizini ve hayatta kalma tahminlemesini kapsamaktadir.

## Tarihsel Arka Plan ve Problem Tanimi

15 Nisan 1912'de, "batmaz" olarak kabul edilen RMS Titanic, ilk seferinde bir buz dagına carparak batmistir. Gemideki 2224 yolcu ve personelden 1502'si hayatini kaybetmistir. Bu facia, hayatta kalma durumunun sadece sansa bagli olmadigini, bazi gruplarin (kadinlar, cocuklar ve ust sinif yolcular) daha yuksek kurtulma oranina sahip oldugunu gostermektedir.

**Hedef:** Yolcu verilerini kullanarak hangi yolcularin shipwreck'ten sag cikabilecegini tahmin eden bir model olusturmaktir.

## Veri Sozlugu (Data Dictionary)

| Degisken | Tanimi | Bilgi |
| :--- | :--- | :--- |
| **Survival** | Hayatta Kalma | 0 = Hayir, 1 = Evet |
| **Pclass** | Bilet Sinifi | 1 = Ust, 2 = Orta, 3 = Alt |
| **Sex** | Cinsiyet | Male / Female |
| **Age** | Yas | Yil bazinda yas |
| **SibSp** | Kardes / Es Sayisi | Titanic'teki kardes, uvey kardes, es sayisi |
| **Parch** | Ebeveyn / Cocuk Sayisi | Titanic'teki anne, baba, cocuk, uvey cocuk sayisi |
| **Ticket** | Bilet Numarasi | Biletin seri numarasi |
| **Fare** | Yolcu Ucreti | Bilet fiyati |
| **Cabin** | Kabin Numarasi | Kabin bilgisi |
| **Embarked** | Binis Limani | C=Cherbourg, Q=Queenstown, S=Southampton |

### Degisken Notlari

- **Pclass:** Sosyo-ekonomik statu (SES) gostergesidir (1st = Ust, 2nd = Orta, 3rd = Alt).
- **Age:** 1 yasindan kucukler ondalikli olarak gosterilmistir. Eger yas tahmin edilmisse xx.5 formatindadir.
- **SibSp:** Kardes (brother, sister, stepbrother, stepsister) ve Es (husband, wife) iliskilerini kapsar. (Nisanlilar ve metresler yok sayilmistir).
- **Parch:** Ebeveyn (mother, father) ve Cocuk (daughter, son, stepdaughter, stepson) iliskilerini kapsar. Sadece dadi ile seyahat eden cocuklar icin Parch=0 sayilmistir.

## Proje Dosyalari

- **train.csv:** Modeli egitmek icin kullanilan hayatta kalma bilgisini (ground truth) iceren 891 yolculuk veri seti.
- **test.csv:** Hayatta kalma bilgisi gizli tutulan, tahminleme yapilacak 418 yolculuk veri seti.
- **gender_submission.csv:** Sadece kadinlarin hayatta kaldigini varsayan ornek bir submission dosyasi.

## Temel Analiz Bulgulari (EDA)

- **Cinsiyet:** Kadin yolcularin kurtulma sansi erkeklere gore cok daha yuksektir.
- **Sinif Etkisi:** 1. sinif yolcularin kurtulma orani en yuksektir.
- **Aile Yapisi:** 2-4 kisilik kucuk aileler en yuksek hayatta kalma basarisini sergilemistir.
- **Ucret (Fare):** Yuksek ucret odeyenler (Ust sinif) daha oncelikli kurtarilmistir.

---
Basari metriki olarak **Accuracy** (Dogruluk yuzdesi) esas alinmaktadir.

