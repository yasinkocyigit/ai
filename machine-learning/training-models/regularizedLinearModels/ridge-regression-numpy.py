import numpy as np


class RidgeRegression:
    def __init__(self, lambda_=1.0):
        """
        lambda_ : regularization strength

        0      -> equivalent to Ordinary Least Squares (OLS)
        large  -> coefficients shrink toward zero
        """
        self.lambda_ = lambda_
        self.intercept_ = None
        self.coef_ = None
        self._beta = None
        self._mu = None
        self._sigma = None

    def _standardize_fit(self, X):
        """
        Compute mean and standard deviation from training data
        and return the standardized features.
        """
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)

        # prevent division by zero for constant columns
        self._sigma[self._sigma == 0] = 1

        return (X - self._mu) / self._sigma

    def _standardize_transform(self, X):
        """
        Standardize new data using training statistics.
        Prevents data leakage.
        """
        return (X - self._mu) / self._sigma

    # main

    def fit(self, X, y):
        """
        Train the model using the closed-form Ridge solution:

            beta = (X^T X + lambda * I)^(-1) X^T y

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            Feature matrix

        y : numpy array of shape (n_samples,)
            Target vector
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        n, p = X.shape

        # standardize features
        X_scaled = self._standardize_fit(X)

        #  add intercept column
        ones = np.ones((n, 1))
        X_b = np.hstack([ones, X_scaled])

        # create regularization matrix
        # Do not penalize the intercept term
        I = np.eye(p + 1)
        I[0, 0] = 0

        # closed-form Ridge solution
        A = X_b.T @ X_b + self.lambda_ * I
        self._beta = np.linalg.inv(A) @ X_b.T @ y

        # store coefficients
        self.intercept_ = self._beta[0]
        self.coef_ = self._beta[1:]

        return self

    def predict(self, X):
        """
        Generate predictions for new samples.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
        """
        X = np.array(X, dtype=float)

        X_scaled = self._standardize_transform(X)

        ones = np.ones((X_scaled.shape[0], 1))
        X_b = np.hstack([ones, X_scaled])

        return X_b @ self._beta

    def score(self, X, y):
        """
        Compute the R² score.

        Returns
        -------
        1.0 : perfect fit
        0.0 : no explanatory power
        <0  : worse than predicting the mean
        """
        y = np.array(y, dtype=float)

        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)

        return 1 - ss_res / ss_tot

    def mse(self, X, y):
        """
        Compute Mean Squared Error (MSE).
        """
        y = np.array(y, dtype=float)

        y_pred = self.predict(X)

        return np.mean((y - y_pred) ** 2)

    def __repr__(self):
        return f"RidgeRegression(lambda_={self.lambda_})"


# example usage

if __name__ == "__main__":

    np.random.seed(42)

    # generate synthetic dataset
    n, p = 100, 3

    X = np.random.randn(n, p)

    true_beta = np.array([2.0, -1.5, 0.8])

    y = X @ true_beta + np.random.randn(n) * 0.5

    # train-test split
    split = int(0.8 * n)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # train Ridge Regression
    model = RidgeRegression(lambda_=1.0)
    model.fit(X_train, y_train)

    print("True coefficients :", true_beta)
    print("Estimated coeffs  :", np.round(model.coef_, 4))
    print("Intercept         :", round(model.intercept_, 4))
    print("Test R²           :", round(model.score(X_test, y_test), 4))
    print("Test MSE          :", round(model.mse(X_test, y_test), 4))

    # effect of different lambda values
    print("\nEffect of Regularization")
    print(f"{'Lambda':>10}  {'R²':>8}  {'MSE':>10}")
    print("-" * 32)

    for lam in [0.0001, 0.1, 1, 10, 100]:
        m = RidgeRegression(lambda_=lam)
        m.fit(X_train, y_train)

        print(
            f"{lam:>10}  "
            f"{m.score(X_test, y_test):>8.4f}  "
            f"{m.mse(X_test, y_test):>10.4f}"
        )