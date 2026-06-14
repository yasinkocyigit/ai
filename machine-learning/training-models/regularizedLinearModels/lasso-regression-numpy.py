import numpy as np

class LassoRegression:
    def __init__(self, lambda_=1.0, max_iter=1000, tol=1e-4):
        """
        Lasso Regression (Coordinate Descent implementation)
        
        lambda_  : regularization strength (L1 penalty)
        max_iter : maximum number of iterations
        tol      : convergence tolerance
        """
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        
        self.coef_ = None
        self.intercept_ = None
        
        self._mu = None
        self._sigma = None

    def _standardize_fit(self, X):
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        self._sigma[self._sigma == 0] = 1.0
        return (X - self._mu) / self._sigma

    def _standardize_transform(self, X):
        return (X - self._mu) / self._sigma

    @staticmethod
    def _soft_threshold(rho, lam):
        """Soft-thresholding operator for L1 regularization"""
        if rho > lam:
            return rho - lam
        elif rho < -lam:
            return rho + lam
        else:
            return 0.0

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n, p = X.shape

        # standardization
        X_scaled = self._standardize_fit(X)
        y_mean = y.mean()
        y_centered = y - y_mean

        # initialize coefficients
        beta = np.zeros(p)

        for iteration in range(self.max_iter):
            beta_old = beta.copy()

            for j in range(p):
                # calculate residual excluding current feature j
                # residual = y - X * beta + Xj * betaj
                residual = y_centered - (X_scaled @ beta) + (X_scaled[:, j] * beta[j])
                
                # compute rho (correlation between feature j and residual)
                rho = (X_scaled[:, j] @ residual) / n
                
                # apply soft thresholding
                beta[j] = self._soft_threshold(rho, self.lambda_)
            
            # convergence check
            if np.max(np.abs(beta - beta_old)) < self.tol:
                break

        self.coef_ = beta
        self.intercept_ = y_mean
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        X_scaled = self._standardize_transform(X)
        return X_scaled @ self.coef_ + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - (ss_res / ss_tot)

if __name__ == "__main__":
    from sklearn.datasets import make_regression

    # generate synthetic data
    X, y = make_regression(n_samples=100, n_features=10, n_informative=3, noise=5, random_state=42)

    # train Lasso
    lasso = LassoRegression(lambda_=0.1)
    lasso.fit(X, y)

    print("Lasso Regression Results:")
    print("-" * 25)
    print(f"Intercept: {lasso.intercept_:.4f}")
    print(f"Coefficients:\n{np.round(lasso.coef_, 4)}")
    print(f"R² Score: {lasso.score(X, y):.4f}")

    # count non-zero coefficients (feature selection)
    non_zero = np.sum(lasso.coef_ != 0)
    print(f"\nNon-zero coefficients: {non_zero} out of 10")
    print("(Lasso eliminated the non-informative features)")
