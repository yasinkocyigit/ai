import numpy as np

class SoftmaxRegression:
    """Multi-class Softmax Regression classifier using Gradient Descent."""
    def __init__(self, learning_rate=0.01, epochs=5000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None  # W matrix of shape (n_classes, n_features + 1) including bias
        self.losses = []

    def _softmax(self, z):
        """Numerically stable Softmax: logits -> probability distribution."""
        # Subtract max value along axis 1 for numerical stability (prevents overflow)
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exps = np.exp(shift_z)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def _one_hot(self, y, K):
        """Convert class labels to one-hot encoded matrix (m, K)."""
        m = len(y)
        one_hot = np.zeros((m, K))
        one_hot[np.arange(m), y] = 1
        return one_hot

    def fit(self, X, y):
        """Train the model on X (m, n) and y (m,) with labels in [0, K-1]."""
        m, n = X.shape
        # Identify the number of unique classes (K)
        unique_classes = np.unique(y)
        K = len(unique_classes)
        
        # Add a column of ones for the bias/intercept term (design matrix X_b)
        X_b = np.c_[np.ones((m, 1)), X]  # Shape: (m, n_features + 1)
        
        # Initialize weights (W) matrix to zeros
        # W has shape (n_classes, n_features + 1)
        self.weights = np.zeros((K, n + 1))
        
        # Convert target labels to 1-hot encoded matrix
        Y_one_hot = self._one_hot(y, K)  # Shape: (m, K)
        self.losses = []

        for epoch in range(self.epochs):
            # Compute linear combinations (logits): shape (m, K)
            logits = np.dot(X_b, self.weights.T)
            
            # Apply softmax to get probability matrix P: shape (m, K)
            P = self._softmax(logits)
            
            # Clip probabilities to avoid log(0) numerical issues
            P = np.clip(P, 1e-15, 1 - 1e-15)

            # Calculate categorical cross-entropy loss
            loss = - (1 / m) * np.sum(Y_one_hot * np.log(P))
            self.losses.append(loss)

            # Calculate gradients with respect to weights: shape (K, n_features + 1)
            # Gradient = (1/m) * (P - Y)^T * X_b
            dw = (1 / m) * np.dot((P - Y_one_hot).T, X_b)

            # Update weights using Gradient Descent
            self.weights -= self.lr * dw

            # Print loss updates
            if epoch % (self.epochs // 10 or 1) == 0:
                print(f"Epoch {epoch}/{self.epochs} - Loss: {loss:.5f}")

        print(f"Training completed. Final Loss: {self.losses[-1]:.5f}")

    def predict_proba(self, X):
        """Return predicted probabilities (m, n_classes) for samples in X."""
        m = X.shape[0]
        # Add bias column
        X_b = np.c_[np.ones((m, 1)), X]
        logits = np.dot(X_b, self.weights.T)
        return self._softmax(logits)

    def predict(self, X):
        """Return predicted class labels (argmax of probabilities)."""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

if __name__ == "__main__":
    print("Generating synthetic 3-class classification data...")
    np.random.seed(42)
    
    # Generate random features (2 features, 300 samples)
    m_samples = 300
    X_dummy = np.random.randn(m_samples, 2)
    
    # Create simple boundaries to assign labels (0, 1, or 2)
    # y = 0 if x1 + x2 < -0.5
    # y = 1 if -0.5 <= x1 + x2 < 0.5
    # y = 2 if x1 + x2 >= 0.5
    scores = X_dummy[:, 0] + X_dummy[:, 1]
    y_dummy = np.zeros(m_samples, dtype=int)
    y_dummy[scores < -0.5] = 0
    y_dummy[(scores >= -0.5) & (scores < 0.5)] = 1
    y_dummy[scores >= 0.5] = 2

    print("Initializing and fitting SoftmaxRegression model...")
    # Feature scaling is highly recommended for faster convergence
    X_scaled = (X_dummy - np.mean(X_dummy, axis=0)) / np.std(X_dummy, axis=0)
    
    model = SoftmaxRegression(learning_rate=0.1, epochs=1000)
    model.fit(X_scaled, y_dummy)

    print("\nLearned Model Weights (shape: n_classes x (n_features + 1)):")
    print(model.weights)

    # Make predictions on test samples
    test_samples = np.array([
        [-1.5, -1.5],  # Should be class 0
        [0.0, 0.0],    # Should be class 1
        [1.5, 1.5]     # Should be class 2
    ])
    # Scale test samples using the same parameters
    test_samples_scaled = (test_samples - np.mean(X_dummy, axis=0)) / np.std(X_dummy, axis=0)
    
    print("\nPredicting on sample test features:")
    print("Features:\n", test_samples)
    print("Predicted Probabilities:\n", model.predict_proba(test_samples_scaled))
    print("Class Predictions:", model.predict(test_samples_scaled))
