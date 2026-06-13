import numpy as np
from sklearn.datasets import make_regression

class CustomElasticNet:
    """
    coordinate descent algoritmasi kullanilarak sifirdan yazilmis elastic net regresyonu
    """
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        """
        alpha    : toplam duzenşilestirme  (matematiksel notasyondaki lambda).
        l1_ratio : L1 ve L2 cezalari arasindaki denge. 
                   1.0 oldugunda LASSO, 0.0 oldugunda Ridge gibi davranir.
        max_iter : maksimum dongu (iterasyon) sayisi.
        tol      : yakinsama toleransi (katsayilar bu degerden az degisiyorsa durur).
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        
        self.coef_ = None
        self.intercept_ = None
        
        self._mu = None
        self._sigma = None

    def _standardize_fit(self, X):
        """veriyi standartlastirma ve parametreleri sakla"""
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        self._sigma[self._sigma == 0] = 1.0
        return (X - self._mu) / self._sigma

    def _standardize_transform(self, X):
        """veriyi standartlastirma - normalizasyon"""
        return (X - self._mu) / self._sigma

    @staticmethod
    def _soft_threshold(z, gamma):
        """
        L1 cezasi icin Soft-Thresholding
        katsayilari sifira ceken matematiksel fonksiyon.
        """
        if z > gamma:
            return z - gamma
        elif z < -gamma:
            return z + gamma
        return 0.0

    def fit_math(self, X, y):
        """
        coordinate descent algoritmasi ile katsayilarin iteratif olarak bulunmasi.
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n, p = X.shape

        
        X_scaled = self._standardize_fit(X)
        y_mean = y.mean()
        y_centered = y - y_mean

        # katsayilari baslat
        beta = np.zeros(p)


        # l1_penalty = alpha * l1_ratio
        # l2_penalty = alpha * (1 - l1_ratio)
        l1_penalty = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1 - self.l1_ratio)

        for iteration in range(self.max_iter):
            beta_old = beta.copy()

            for j in range(p):
                # j. katsayi haric diger ozelliklerin tahmin uzerindeki etkisini cikar
                residual = y_centered - (X_scaled @ beta) + (X_scaled[:, j] * beta[j])

                # rho / n
                rho = (X_scaled[:, j] @ residual) / n

                # Elastic Net katsayi guncelleme kurali:
                # Pay: L1 etkisi (Soft-Thresholding)
                # Payda: L2 etkisi (Varyans + l2_penalty)
                
                variance = (X_scaled[:, j] ** 2).sum() / n
                
                beta[j] = self._soft_threshold(rho, l1_penalty) / (variance + l2_penalty)

            # convergence kontrolu
            max_diff = np.max(np.abs(beta - beta_old))
            if max_diff < self.tol:
                break

        self.coef_ = beta
        self.intercept_ = y_mean

        return self

    def predict(self, X):
        """ogrenilen katsayilarla tahminler yapma"""
        X = np.array(X, dtype=float)
        X_scaled = self._standardize_transform(X)
        return X_scaled @ self.coef_ + self.intercept_

    def score(self, X, y):
        """R^2"""
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - (ss_res / ss_tot)


if __name__ == "__main__":
    from sklearn.linear_model import ElasticNet

    np.random.seed(42)

    # generate data - n=100, p=5
    X, y = make_regression(n_samples=100, n_features=5, noise=10.0, random_state=42)

    X[:, 2] = X[:, 1] * 0.8 + np.random.randn(100) * 0.1

    # params
    alpha_val = 0.5
    l1_ratio_val = 0.5

    # model
    custom_model = CustomElasticNet(alpha=alpha_val, l1_ratio=l1_ratio_val)
    custom_model.fit_math(X, y)
    
    # model - Scikit-learn
    # Scikit-learn icin olceklendirme yapma
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X_scaled = (X - mu) / sigma
    
    sklearn_model = ElasticNet(alpha=alpha_val, l1_ratio=l1_ratio_val, fit_intercept=True)
    sklearn_model.fit(X_scaled, y)

    print("--- katsayi karsilastirma ---")
    print(f"matematiksel model karsilastirma : {np.round(custom_model.coef_, 4)}")
    print(f"Scikit-learn katsayilari       : {np.round(sklearn_model.coef_, 4)}\n")
    
    print("--- intercept ---")
    print(f"matematiksel model : {custom_model.intercept_:.4f}")
    print(f"Scikit-learn       : {sklearn_model.intercept_:.4f}\n")