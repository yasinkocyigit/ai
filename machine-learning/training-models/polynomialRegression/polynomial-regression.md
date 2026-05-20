# Polynomial Regression (Polinom Regresyonu) Teorik El Kitabı

Bu doküman, doğrusal olmayan veri yapılarını modellemek için kullanılan Polinom Regresyonu algoritmasının temel mantığını, matematiksel altyapısını ve mülakatlarda karşına çıkabilecek kritik kavramlarını özetler. Kod uygulamaları ve pratik örnekler `.ipynb` dosyası içerisinde yer almaktadır.

---

## 1. Temel Mantık: Doğrusal Modeli "Kandırmak"

Doğrusal Regresyon (Linear Regression) özünde sadece düz bir çizgi çizebilir. Eğer verileriniz eğrisel (curvilinear) bir yapıya sahipse, düz bir çizgi çekmek yüksek tahmin hatalarına (underfitting) neden olur.

Polinom Regresyonunun temel sırrı, **modelin algoritmasını değiştirmek değil, modelin önüne koyduğumuz veri matrisini bükmektir.**

### Çalışma Prensibi
1. Elimizde tek bir özellik (feature) olduğunu varsayalım: $x$
2. Veri setine yeni bir sütun açıyoruz ve mevcut $x$ değerlerinin karesini ($x^2$), küpünü ($x^3$) vb. alıp buraya yazıyoruz. Buna **Özellik Genişletme (Feature Expansion)** denir.
3. Standart Doğrusal Regresyon modeline bu genişletilmiş matrisi veriyoruz. Model hala kendi içinde şu düz, toplamsal denklemi çözdüğünü sanıyor:
   $$ y = \theta_0 + \theta_1 \cdot x_1 + \theta_2 \cdot x_2 $$
4. Biz modelin içine $x_2$ gördüğü yere verinin karesini ($x^2$) gönderdiğimiz için denklem dışarıdan bakıldığında bir parabol eğrisine dönüşüyor:
   $$ y = \theta_0 + \theta_1 \cdot x + \theta_2 \cdot x^2 $$

Algoritma arka planda doğrusal (linear) katsayı matematiği çalıştırırken, grafiğe çizdirdiğimizde veriye tam oturan bir polinom eğrisi elde etmiş oluyoruz.

---

## 2. Matematiksel Genel Denklem

Polinom regresyonu, bağımsız değişkenin üstel güçlerini sisteme dahil ederek denklemi $n$. dereceye kadar genişletir:

$$ y = \beta_0 + \beta_1 \cdot x + \beta_2 \cdot x^2 + \beta_3 \cdot x^3 + \dots + \beta_n \cdot x^n + \epsilon $$

*   **$y$:** Bağımlı değişken (Hedef / Target) - *Örn: Maaş veya Basınç*
*   **$x$:** Bağımsız değişken (Özellik / Feature) - *Örn: Deneyim Yılı veya Sıcaklık*
*   **$\beta_0$:** Kesişim noktası (Bias / Intercept)
*   **$\beta_1, \beta_2, \dots, \beta_n$:** Modelin optimizasyon sürecinde öğreneceği ağırlıklar (Coefficients)
*   **$n$:** Polinomun derecesi (Degree)
*   **$\epsilon$:** Hata terimi (Residual / Error)

---

## 3. Kritik Dengeler: Bias-Variance Dilemması

Polinom regresyonunda en önemli hiperparametre doğru derece ($n$) seçimidir. Modelin esnekliği bu dereceye bağlıdır.

*   **Underfitting (Yüksek Bias):** Polinom derecesinin verinin karmaşıklığına kıyasla çok düşük seçilmesi durumudur (Örn: Eğri veriye 1. dereceden düz çizgi çekmek). Model verideki temel yapıyı ve kıvrımları kavrayamaz.
*   **Overfitting (Yüksek Varyans):** Polinom derecesinin çok yüksek seçilmesi durumudur (Örn: Dereceyi 10 veya 20 yapmak). Model, eğitim verisindeki rastgele salınımları ve gürültüleri (noise) bile ezberler. Grafik deli gibi zikzaklar çizer. Eğitim setinde hata sıfıra yaklaşsa da model yeni (test) verilerde tamamen çuvallar.

### Kombinasyon Patlaması (Combinatorial Explosion)
Veri setinde birden fazla özellik olduğunda (Örn: $a$ ve $b$ özellikleri), `PolynomialFeatures` sınıfı sadece özelliklerin kendi güçlerini ($a^2, b^2$) eklemez; aynı zamanda belirlenen dereceye kadar olan **tüm etkileşim kombinasyonlarını ($ab, a^2b, ab^2$)** da matrise ekler. 

Bu durum özellik sayısının aşırı derecede, katlanarak büyümesine (Combinatorial Explosion) neden olur. Çok özellikli veri setlerinde yüksek dereceli polinomlar seçerken bellek (RAM) ve overfitting riskine karşı dikkatli olunmalıdır.

---

## 4. Avantajlar ve Dezavantajlar Özeti

### Avantajları
*   Doğrusal olmayan, karmaşık ve eğrisel ilişkileri başarıyla modeller.
*   Doğrusal regresyonun o güçlü, kararlı ve hızlı analitik çözüm (Least Squares / En Küçük Kareler) altyapısını aynen kullanmaya devam eder.

### Dezavantajları
*   **Aykırı Değer (Outlier) Hassasiyeti:** Aykırı değerlere karşı aşırı hassastır; uçtaki tek bir veri noktası yüksek üslü terimlerin etkisiyle eğrinin yönünü tamamen saptırabilir.
*   **Ekstrapolasyon Tehlikesi:** Verinin bittiği sınırların dışındaki alanlar için (geleceğe yönelik) tahmin yaparken eğri aşağı veya yukarı doğru çok sert kırılır. Bu durum sınır dışı tahminleri tamamen kararsız ve güvenilmez hale getirir.