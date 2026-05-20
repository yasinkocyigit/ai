# Gradient Descent

Gradient Descent, bir maliyet fonksiyonunu minimize etmek için parametreleri iteratif olarak güncelleyen genel amaçlı bir optimizasyon algoritmasıdır.

## Temel Fikir

Parametre vektoru `θ`'yi rastgele bir noktadan başlatıp, maliyet fonksiyonunun gradyanının ters yönünde adım adım ilerleyerek minimuma ulaşmaya çalışırsın. Gradyan sıfıra ulaştığında minimuma gelmiş olursun.

```
θ_next = θ - η * ∇MSE(θ)
```

- `η` (eta): learning rate, adım büyüklüğünü belirler
- `∇MSE(θ)`: maliyet fonksiyonunun gradyanı

---

## Learning Rate

| Durum | Sonuç |
|---|---|
| Çok küçük | Yavaş yakınsar, çok iterasyon gerekir |
| Çok büyük | Minimumu atlayabilir, ıraksayabilir |
| Uygun | Birkaç iterasyonda yakınsar |

---

## Karşılaşılan Sorunlar

- **Local minimum**: Rastgele başlangıç noktası kötü seçilirse global minimum yerine local minimuma takılabilir.
- **Plateau**: Düz bölgeler geçişi yavaşlatır.
- **Feature scaling**: Özellikler farklı ölçeklerdeyse gradient descent yavaşlar. `StandardScaler` kullanmak bu sorunu çözer.

Linear Regression için MSE maliyet fonksiyonu **convex** olduğundan local minimum sorunu yoktur, her zaman global minimuma yakınsanır.

---

## Üç Varyant

### 1. Batch Gradient Descent

Her adımda tüm veri setiyle gradyanı hesaplar.

```python
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
```

- Büyük veri setlerinde çok yavaş
- Sabit learning rate ile convex fonksiyonlarda global minimuma garantili yakınsama
- Özellik sayısı çok olduğunda Normal Denklem'den çok daha hızlı

---

### 2. Stochastic Gradient Descent (SGD)

Her adımda tek rastgele bir örnek seçip gradyanı hesaplar.

```python
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
```

- Çok hızlı, büyük veri setlerine uygun (out-of-core destekler)
- Maliyet fonksiyonu düzensiz iner, minimumda salınım yapar
- Local minimumdan kaçmakta iyidir
- **Simulated annealing**: Learning rate'i zamanla azaltarak salınımı bastırabilirsin

---

### 3. Mini-batch Gradient Descent

Her adımda küçük rastgele gruplarla (mini-batch) gradyanı hesaplar.

- GPU optimizasyonundan faydalanır
- SGD'den daha stabil, Batch GD'den daha hızlı
- En yaygın kullanılan varyant

---

## Algoritmaların Karşılaştırması

| Algoritma | Büyük m | Out-of-core | Büyük n | Scaling |
|---|---|---|---|---|
| Normal Equation | Yavaş | Hayır | Yavaş | Hayır |
| Batch GD | Yavaş | Hayır | Hızlı | Gerekli |
| Stochastic GD | Hızlı | Evet | Hızlı | Gerekli |
| Mini-batch GD | Hızlı | Evet | Hızlı | Gerekli |

---

## Scikit-Learn ile SGD

```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
```

---

## Notlar

- Tüm varyantlar eğitim sonunda benzer modeller üretir.
- Feature scaling (örn. `StandardScaler`) kullanmak yakınsama hızını ciddi artırır.
- Tolerans (`ε`) yöntemi: gradient vektörünün normu küçük bir eşiğin altına düşünce algoritmayı durdur.