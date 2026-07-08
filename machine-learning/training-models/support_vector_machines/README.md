# Destek Vektör Makineleri - Support Vector Machines - SVM

Destek Vektör Makineleri algoritmasının doğrusal, doğrusal olmayan, sınıflandırma ve regresyon ile optimizasyon çözümleri yer almaktadır.

## Klasör Yapısı ve İçerik

Klasör, dört ana alt bölüme ayrılmıştır:

1. **[scriptsSVM](./scriptsSVM/)**
   - **[SVM_From_Scratch.ipynb](./scriptsSVM/SVM_From_Scratch.ipynb)**: Primal formdan Lagrangian fonksiyonuna geçiş, kısmi türevlerin durağanlık koşulu gereği sıfıra eşitlenmesi ve dual probleme dönüşüm adımları.
   - CVXPY çözücüsü kullanılarak Hard Margin, Soft Margin ve Polinom Kernel SVM modellerinin optimizasyonu.
   - Sıfırdan SMO - Sequential Minimal Optimization - algoritmasının Numpy ile kodlanması ve KKT koşulları.

2. **[LinearSVMClassification](./LinearSVMClassification/)**
   - **[Linear_SVM_Classification.ipynb](./LinearSVMClassification/Linear_SVM_Classification.ipynb)**: Elle sınır hesaplama örneği.
   - Karar doğrusunun Matplotlib ile çizimi için gereken cebirsel dönüşüm.
   - Primal SVM'in genel Karesel Programlama - Quadratic Programming - QP - matris gösterimi ($H, f, A, b$).
   - Hinge Loss ve Scikit-Learn `LinearSVC` ile online öğrenme sağlayan `SGDClassifier`.

3. **[NonlinearSVMClassification](./NonlinearSVMClassification/)**
   - **[Nonlinear_SVM_Classification.ipynb](./NonlinearSVMClassification/Nonlinear_SVM_Classification.ipynb)**: Kernel Trick
   - Polinom çekirdeği katsayıları (`degree`, `coef0`).
   - Gaussian RBF benzerlik özellikleri, data point'in bir boyuttan, iki boyutlu uzaya taşınması ve Scikit-Learn uygulaması.

4. **[SVMRegression](./SVMRegression/)**
   - **[SVM_Regression.ipynb](./SVMRegression/SVM_Regression.ipynb)**: Sınıflandırma ve regresyon modelleri arasındaki mantıksal farklar.
   - Epsilon ($\epsilon$) duyarsızlık hiperparametresi ve $\xi, \xi^*$ slack değişkenlerinin çalışma mantığı.
   - Scikit-Learn `LinearSVR` ile sınır çizgilerinin görselleştirilmesi.

## Görselleştirmeler
Modellerin karar sınırları, marjları ve destek vektörleri grafikler üzerinde çizilmiştir. Özellikle esnek marj modelinde marj ihlali yapan ($\\xi_i > 0$) noktalar halkalarla belirtilmiş ve bu noktaların marj sınırlarına olan dikey uzaklıkları - slack vektörleri - kesikli doğrularla gösterilmiştir.

## Gereksinimler
Kodların çalıştırılabilmesi için gerekli kütüphaneler:
- `numpy`
- `cvxpy`
- `matplotlib`
- `scikit-learn`
