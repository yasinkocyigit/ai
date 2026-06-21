import numpy as np

class LogisticRegression:
    """
    A simple binary logistic regression classifier implemented from scratch using NumPy.
    Uses Gradient Descent for optimization.
    """
    def __init__(self, learning_rate=0.001, epochs=5000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, z):
        """
        Sigmoid activation function mapping any real value to the range [0, 1].
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Fit the logistic regression model on training data X and targets y.
        
        Parameters:
        X (numpy.ndarray): Training features of shape (m_samples, n_features)
        y (numpy.ndarray): Binary target labels of shape (m_samples,) or (m_samples, 1)
        """
        m, n = X.shape
        # agirliklarin ve sabit terimin sifir olarak ilklendirilmesi
        self.weights = np.zeros(n)
        self.bias = 0.0
        
        # vektor islemlerinin dogrulugu icin hedef degisken boyutunun duzenlenmesi
        y = y.squeeze()
        self.losses = []

        for epoch in range(self.epochs):
            # dogrusal tahminin hesaplanmasi (girdilerin katsayilar ve sabit terim ile dogrusal birlesimi)
            linear_model = np.dot(X, self.weights) + self.bias
            # olasilik tahminlerini elde etmek icin sigmoid aktivasyon fonksiyonunun uygulanmasi
            y_pred = self._sigmoid(linear_model)

            # kayip hesabi sirasinda sayisal kararlilik icin sifira bolmenin veya log(0) durumunun engellenmesi
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

            # ikili capraz entropi kaybinin (log loss) hesaplanmasi
            loss = - (1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            self.losses.append(loss)

            # katsayilara ve sabit terime gore gradyanlarin hesaplanmasi
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            # gradyan inisi guncelleme kurali ile katsayilarin ve sabit terimin guncellenmesi
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # belirli araliklarla kayip degerinin ekrana yazdirilmasi
            if epoch % (self.epochs // 10 or 1) == 0:
                print(f"Epoch {epoch}/{self.epochs} - Loss: {loss:.5f}")

        print(f"Training completed. Final Loss: {self.losses[-1]:.5f}")

    def predict_proba(self, X):
        """
        Predict probability estimates for each sample in X.
        
        Returns:
        numpy.ndarray: Predicted probabilities of class 1
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        Predict binary class labels (0 or 1) for samples in X.
        
        Parameters:
        X (numpy.ndarray): Input features
        threshold (float): Decision threshold (default: 0.5)
        
        Returns:
        numpy.ndarray: Binary predictions (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

# kullanim ornegini gostermek icin basit calistirma blogu
if __name__ == "__main__":
    print("Generating synthetic binary classification data...")
    # tekrar uretilebilirlik icin rastgelelik tohumunun sabitlenmesi
    np.random.seed(42)
    
    # uc oznitelikli 200 rastgele ornek veri uretilmesi
    m_samples, n_features = 200, 3
    X_dummy = np.random.randn(m_samples, n_features)
    
    # rastgele gercek katsayilarin ve sabit terimin tanimlanmasi
    true_weights = np.array([2.5, -1.2, 0.8])
    true_bias = -0.5
    
    # gercek log-oranlarin uretilmesi ve binom dagilimi ile hedef siniflarin olusturulmasi
    log_odds = np.dot(X_dummy, true_weights) + true_bias
    probs = 1 / (1 + np.exp(-log_odds))
    y_dummy = np.random.binomial(1, probs, m_samples)

    print("Initializing and fitting LogisticRegression model...")
    model = LogisticRegression(learning_rate=0.1, epochs=1000)
    model.fit(X_dummy, y_dummy)

    print("\nLearned Model Parameters:")
    print("Intercept (bias):", model.bias)
    print("Coefficients (weights):", model.weights)
    print("True Coefficients:   ", true_weights)

    # test ornekleri uzerinde tahminlerin gerceklestirilmesi
    test_sample = np.array([[1.0, -1.0, 0.5], [-1.0, 1.0, -0.5]])
    print("\nPredicting on sample test features:")
    print("Features:\n", test_sample)
    print("Predicted Probabilities:", model.predict_proba(test_sample))
    print("Class Predictions (threshold=0.5):", model.predict(test_sample))
