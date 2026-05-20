# Model Performans Analizi: Öğrenme Eğrileri ve Bias-Varyans Dengesi

Bir Makine Öğrenimi modelinin tahmin başarısını artırmak ve ezberleme/öğrenememe problemlerini teşhis etmek için kullanılan evrensel metrikler ve hata analiz yöntemleri aşağıda özetlenmiştir.

---

## 1. Hatanın Üç Bileşeni (Bias, Varyans ve İndirgenemez Hata)

İstatistik ve Makine Öğrenimi teorisine göre, bir modelin genelleme hatası (generalization error) üç farklı ve bağımsız bileşenin toplamından oluşur:

### Bias (Yanlılık)
Modelin veri setini aşırı basitleştirmesinden ve yanlış varsayımlar yapmasından kaynaklanır. 
*   **Tanım:** Modelin ortalama tahmini ile gerçek değerler arasındaki farktır. 
*   **Karakteristik:** Doğrusal Regresyon gibi algoritmalar eğrisel verilere uygulandığında yüksek bias üretir. Öğrenmesi hızlıdır ancak karmaşık örüntüleri yakalayamaz.
*   **Sonuç:** Yüksek bias, modelin hem eğitim hem de test setinde kötü performans göstermesine, yani **Underfitting (Yetersiz Öğrenme)** problemine yol açar.

### Varyans (Variance)
Modelin eğitim verisindeki çok küçük dalgalanmalara ve gürültülere karşı gösterdiği aşırı hassasiyettir.
*   **Tanım:** Eğitim verisi değiştirildiğinde, modelin yapacağı tahminlerin ne kadar değişeceğinin (sapacağının) ölçüsüdür.
*   **Karakteristik:** Karar Ağaçları (Decision Trees), KNN veya yüksek dereceli Polinom Regresyon modelleri yüksek serbestlik derecesine sahiptir ve yüksek varyans üretmeye eğilimlidir.
*   **Sonuç:** Model veri setindeki gürültüleri ezberlediği için **Overfitting (Aşırı Öğrenme)** problemine yol açar.

### İndirgenemez Hata (Irreducible Error)
Doğrudan verinin kendi içindeki gürültüden (bozuk sensörler, eksik veriler, hatalı etiketlemeler) kaynaklanır. Modeli değiştirerek veya hiperparametre ayarlayarak düzeltilemez; tek çözüm veriyi temizlemektir.

---

## 2. Bias - Varyans Dengesi (The Trade-off)

İdeal bir makine öğrenimi modelinden beklenen, hem düşük bias hem de düşük varyans değerlerine sahip olmasıdır. Ancak matematikte bu iki metrik birbiriyle ters orantılı çalışır:

*   **Modelin karmaşıklığı artırıldığında:** Varyans artar, bias düşer.
*   **Modelin karmaşıklığı azaltıldığında:** Bias artar, varyans düşer.

Veri bilimi projelerindeki temel amaç, toplam hatayı minimize edecek bu optimum denge (trade-off) noktasını bulmaktır.

---

## 3. Öğrenme Eğrileri (Learning Curves) ile Teşhis

Öğrenme eğrileri, doğru Bias-Varyans dengesini bulmak ve veri miktarının modele etkisini görmek için kullanılan en güçlü teşhis aracıdır. Eğitim seti boyutu (X ekseni) arttıkça, modelin eğitim ve doğrulama (validation) setleri üzerindeki skorlarının (Y ekseni) nasıl değiştiğini gösterir.

### Underfitting (Yüksek Bias) Eğri Karakteristiği
*   Eğitim skorları (veya hataları) belli bir noktadan sonra sabitlenir (plato çizer).
*   Çapraz doğrulama (cross-validation) skorları da ona çok yakın bir seviyede yataya bağlar.
*   **Teşhis:** İki eğri birbirine çok yakınlaşır ancak modelin genel başarı skoru düşüktür.
*   **Çözüm:** Modelin kapasitesi yetersizdir. Bu senaryoda **modele daha fazla veri eklemek hiçbir işe yaramaz.** Daha karmaşık bir algoritma seçmek veya özellik mühendisliği (feature engineering) yapmak gerekir.

### Overfitting (Yüksek Varyans) Eğri Karakteristiği
*   Model eğitim verisini ezberlediği için eğitim skoru (accuracy) mükemmele yakındır.
*   Ancak doğrulama (validation) skoru çok daha düşük kalır.
*   **Teşhis:** İki eğri arasında bariz ve geniş bir **boşluk (gap)** oluşur.
*   **Çözüm:** Bu model yeni verilere genelleme yapamamaktadır. Eğrileri birbirine yaklaştırmak için **modele daha fazla eğitim verisi sağlamak (veri setini büyütmek)** veya modelin karmaşıklığını kısıtlamak (regularization) gerekir.

---

## 4. Scikit-Learn İle Pratik Uygulama Notu
Scikit-Learn kütüphanesi, öğrenme eğrilerini manuel döngülerle (for loop) hesaplamak yerine tek satırda çözüm sunan `learning_curve` modülüne sahiptir. Bu modül `sklearn.model_selection` içinden çağrılır ve belirtilen K-Fold (Çapraz Doğrulama) değeriyle modelin tüm iterasyon skorlarını otomatik olarak döndürür.