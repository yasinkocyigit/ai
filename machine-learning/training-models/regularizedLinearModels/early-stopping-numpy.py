import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class SGDRegressorWithEarlyStopping:
    def __init__(self, eta0=0.01, n_epochs=1000, patience=5):
        """
        Gradient Descent with Early Stopping
        
        eta0     : learning rate
        n_epochs : maximum number of iterations
        patience : number of epochs to wait for improvement before stopping
        """
        self.eta0 = eta0
        self.n_epochs = n_epochs
        self.patience = patience
        
        self.weights = None
        self.best_weights = None
        self.best_epoch = None
        
        self.train_errors = []
        self.val_errors = []

    def fit(self, X_train, y_train, X_val, y_val):
        n_samples, n_features = X_train.shape
        
        # add bias term
        X_train_b = np.c_[np.ones((n_samples, 1)), X_train]
        X_val_b = np.c_[np.ones((len(X_val), 1)), X_val]
        
        # init weights
        self.weights = np.random.randn(n_features + 1, 1)
        
        best_val_error = float("inf")
        wait = 0
        
        for epoch in range(self.n_epochs):
            # compute gradients
            gradients = 2/n_samples * X_train_b.T @ (X_train_b @ self.weights - y_train)
            
            # update weights
            self.weights = self.weights - self.eta0 * gradients
            
            # track errors
            y_train_pred = X_train_b @ self.weights
            y_val_pred = X_val_b @ self.weights
            
            train_error = mean_squared_error(y_train, y_train_pred)
            val_error = mean_squared_error(y_val, y_val_pred)
            
            self.train_errors.append(train_error)
            self.val_errors.append(val_error)
            
            # Early Stopping Logic
            if val_error < best_val_error:
                best_val_error = val_error
                self.best_weights = self.weights.copy()
                self.best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"Stopping early at epoch {epoch}")
                    break
        
        # restore best weights
        self.weights = self.best_weights
        return self

    def predict(self, X):
        X_b = np.c_[np.ones((len(X), 1)), X]
        return X_b @ self.weights

if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate synthetic data (Quadratic + Noise)
    m = 200
    X = 6 * np.random.rand(m, 1) - 3
    y = 2 + X + 0.5 * X**2 + np.random.randn(m, 1)
    
    # Split
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature Scaling (essential for Gradient Descent)
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    
    # Use high degree polynomial to encourage overfitting
    poly = PolynomialFeatures(degree=20, include_bias=False)
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(poly.fit_transform(X_train_raw))
    X_val = scaler.transform(poly.transform(X_val_raw))
    
    # Train
    model = SGDRegressorWithEarlyStopping(eta0=0.1, n_epochs=2000, patience=10)
    model.fit(X_train, y_train, X_val, y_val)
    
    print(f"Best epoch: {model.best_epoch}")
    
    # Plot
    plt.plot(np.sqrt(model.train_errors), "r--", label="Training Set")
    plt.plot(np.sqrt(model.val_errors), "b-", label="Validation Set")
    plt.axvline(x=model.best_epoch, color='g', linestyle=':', label="Early Stopping Point")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Early Stopping Demonstration")
    plt.show()
