# Machine Learning

Makine öğrenmesi algoritmalarını öğrenmek, uygulamak ve ileride referans olarak kullanmak için hazırlanmış notebooklar.

Her notebook: teori + formül + sıfırdan implementasyon + sklearn ile implementasyon içerir.

---

## Dosya Yapısı

```text
.
├── 00-references
├── 01-fundamentals
├── 02-linear-models
├── 03-classification
├── 04-svm
├── 05-decision-trees
├── 06-ensemble-methods
├── 07-unsupervised
├── 08-model-evaluation
└── 09-projects
```

## Dizin Yapısı

### 00-references
Lazım olduğunda bakılacak referans notlar.
- `formulas.ipynb` - tüm önemli formüller bir arada
- `when-to-use-what.ipynb` - hangi algoritmayı hangi durumda kullan
- `notes.ipynb` - öğrenirken alınan genel notlar

### 01-fundamentals
ML'e giriş ve gerçek bir projenin baştan sona nasıl yürütüldüğü.
- `ml-landscape.ipynb` - ML nedir, öğrenme türleri, temel kavramlar
- `end-to-end-project.ipynb` - veri toplama, temizleme, model seçimi, deploy

### 02-linear-models
Doğrusal modeller ve regresyon ailesi.
- `linear-regression.ipynb` - en temel model, gradient descent burada kavranır
- `polynomial-regression.ipynb` - doğrusal olmayan ilişkiler
- `logistic-regression.ipynb` - sınıflandırmaya giriş
- `regularization.ipynb` - Ridge, Lasso, ElasticNet

### 03-classification
Sınıflandırma problemleri ve performans ölçümü.
- `binary-classification.ipynb` - iki sınıflı problemler
- `multiclass-classification.ipynb` - çok sınıflı problemler
- `performance-metrics.ipynb` - confusion matrix, precision, recall, F1, ROC

### 04-svm
Destek vektör makineleri.
- `linear-svm.ipynb` - doğrusal ayrılabilir problemler
- `nonlinear-svm.ipynb` - kernel trick, RBF
- `svm-regression.ipynb` - SVR

### 05-decision-trees
Karar ağaçları.
- `decision-trees.ipynb` - CART algoritması, Gini, entropy, görselleştirme

### 06-ensemble-methods
Birden fazla modeli bir araya getiren yöntemler.
- `voting-classifiers.ipynb` - hard ve soft voting
- `bagging-pasting.ipynb` - örnekleme ile çeşitlilik
- `random-forest.ipynb` - en çok kullanılan ensemble yöntemi
- `adaboost.ipynb` - boosting'e giriş
- `gradient-boosting.ipynb` - XGBoost, LightGBM

### 07-unsupervised
Etiketlenmemiş veri ile öğrenme.
- `pca.ipynb` - boyut indirgeme, varyans koruma
- `kernel-pca.ipynb` - doğrusal olmayan boyut indirgeme
- `dimensionality-reduction.ipynb` - LLE ve diğer teknikler

### 08-model-evaluation
Model seçimi ve iyileştirme.
- `cross-validation.ipynb` - k-fold, stratified
- `grid-search.ipynb` - hyperparameter optimizasyonu
- `learning-curves.ipynb` - overfitting/underfitting tespiti

### 09-projects
Gerçek verisetleri üzerinde uçtan uca projeler.

## Kullanılan Kütüphaneler
- numpy, pandas
- scikit-learn
- matplotlib, seaborn
- xgboost, lightgbm

---

