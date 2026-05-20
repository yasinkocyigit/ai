# Lineer Regresyon — Çalışma Notları

> Kaynak: *Hands-On Machine Learning* — Bölüm 4

---

## 1. Model Nedir?

Lineer regresyon, giriş özelliklerinin ağırlıklı toplamı artı bir bias terimi hesaplayarak tahmin üretir.

### Genel Formül

$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$$

| Sembol | Anlamı |
|--------|--------|
| $\hat{y}$ | Tahmin edilen değer |
| $n$ | Özellik (feature) sayısı |
| $x_i$ | i'inci özelliğin değeri |
| $\theta_j$ | j'inci model parametresi ($\theta_0$ = bias) |

### Vektör Formu

$$\hat{y} = h_\theta(\mathbf{x}) = \boldsymbol{\theta}^T \cdot \mathbf{x}$$

- $\boldsymbol{\theta}$ — bias + ağırlıkları içeren parametre vektörü
- $\mathbf{x}$ — özellik vektörü ($x_0 = 1$ sabit)
- $\boldsymbol{\theta}^T \cdot \mathbf{x}$ — nokta çarpımı (dot product)

---

## 2. Eğitim: Maliyet Fonksiyonu

Model eğitmek = $\boldsymbol{\theta}$ parametrelerini en iyi sonucu verecek şekilde ayarlamak.

### MSE (Mean Squared Error)

$$\text{MSE}(\boldsymbol{\theta}) = \frac{1}{m} \sum_{i=1}^{m} \left( \boldsymbol{\theta}^T \cdot \mathbf{x}^{(i)} - y^{(i)} \right)^2$$

- $m$ — eğitim örneği sayısı
- RMSE minimize etmek yerine **MSE minimize edilir** (sonuç aynı, hesaplama daha kolay)

---

## 3. Normal Denklem (Kapalı Form Çözüm)

$\boldsymbol{\theta}$'yı bulmak için türetilmiş matematiksel formül:

$$\hat{\boldsymbol{\theta}} = \left( \mathbf{X}^T \mathbf{X} \right)^{-1} \mathbf{X}^T \mathbf{y}$$

### NumPy Uygulaması

```python
import numpy as np

# Veri üret
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Bias için x0=1 sütunu ekle
X_b = np.c_[np.ones((100, 1)), X]

# Normal Denklem
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# Beklenen: theta_0 ≈ 4, theta_1 ≈ 3

# Tahmin
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
```

### Scikit-Learn ile Aynı İşlem

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

print(lin_reg.intercept_)  # theta_0 (bias)
print(lin_reg.coef_)       # theta_1, theta_2, ...
lin_reg.predict(X_new)
```

---

## 4. Hesaplama Karmaşıklığı

| Durum | Karmaşıklık | Not |
|-------|-------------|-----|
| Özellik sayısına göre | $O(n^{2.4})$ – $O(n^3)$ | $n$ büyüyünce çok yavaşlar |
| Örnek sayısına göre | $O(m)$ | Büyük veri setinde verimli |
| Tahmin yapma | $O(m \cdot n)$ | Doğrusal, hızlı |

> **Dikkat:** $n \geq 100.000$ özellik varsa Normal Denklem yavaşlar. Bu durumda **Gradient Descent** kullanılmalı (sonraki konu).

---

## 5. PyTorch ile Lineer Regresyon

PyTorch'ta iki farklı yaklaşım var: **manuel** (sıfırdan) ve **nn.Linear** (hazır katman).

### 5.1 Manuel Yaklaşım (Gradient Descent)

Parametre güncelleme kuralı:

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \cdot \nabla_{\boldsymbol{\theta}}\, \text{MSE}(\boldsymbol{\theta})$$

```python
import torch

torch.manual_seed(42)
X = 2 * torch.rand(100, 1)
y = 4 + 3 * X + torch.randn(100, 1)

# requires_grad=True -> autograd aktif
theta = torch.randn(2, 1, requires_grad=True)

X_b = torch.cat([torch.ones(100, 1), X], dim=1)

lr = 0.01
for epoch in range(1000):
    y_pred = X_b @ theta                    # (100,2) x (2,1) -> (100,1)
    loss = ((y_pred - y) ** 2).mean()       # MSE

    loss.backward()                         # gradyanlari hesapla

    with torch.no_grad():
        theta -= lr * theta.grad            # parametre guncelle
        theta.grad.zero_()                  # gradyani sifirla

print(theta)  # theta_0 ≈ 4, theta_1 ≈ 3
```

### 5.2 nn.Linear ile (Önerilen Yol)

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
X = 2 * torch.rand(100, 1)
y = 4 + 3 * X + torch.randn(100, 1)

# 1 giris -> 1 cikis (bias=True varsayilan)
model = nn.Linear(in_features=1, out_features=1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(X)                       # forward pass
    loss = criterion(y_pred, y)             # MSE hesapla

    optimizer.zero_grad()                   # gradyanlari sifirla
    loss.backward()                         # geri yayilim
    optimizer.step()                        # agirliklari guncelle

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

print("theta_0 (bias):", model.bias.item())     # ≈ 4
print("theta_1 (weight):", model.weight.item()) # ≈ 3

with torch.no_grad():
    print(model(torch.tensor([[0.0], [2.0]])))
```

### 5.3 Üç Yöntemin Karşılaştırması

| | NumPy (Normal Denklem) | Scikit-Learn | PyTorch |
|---|---|---|---|
| Yöntem | Kapalı form | Kapalı form | Gradient Descent |
| İterasyon | Yok | Yok | Var |
| GPU desteği | Hayır | Hayır | Evet |
| Büyük veri | Yavaş $O(n^3)$ | Yavaş | Verimli |
| Derin öğrenmeye geçiş | Hayır | Hayır | Doğal |

> **Ne zaman PyTorch?** Lineer regresyon için overkill olsa da, derin öğrenme iş akışını öğrenmek için iyi başlangıç noktası. `forward -> loss -> backward -> step` döngüsü tüm sinir ağlarında aynıdır.

---

## 6. Özet

- Lineer regresyon = ağırlıklı toplam + bias
- Eğitim hedefi = MSE'yi minimize et
- Normal Denklem = kapalı form, iterasyon yok
- Az özellik → Normal Denklem; çok özellik → Gradient Descent
- Scikit-Learn `LinearRegression` arka planda benzer hesaplamayı yapar
- PyTorch temel döngü: `forward -> loss -> backward -> step`
- `requires_grad=True` → autograd gradyanı otomatik hesaplar
- `optimizer.zero_grad()` her iterasyonda çağrılmalı

---

*Son güncelleme: Mayıs 2026*