import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles


def generate_nonlinear_data():
    X, y = make_circles(n_samples=100, factor=0.3, noise=0.1)
    y = np.where(y <= 0, -1, 1)
    return (X, y)


def polynomial_kernel(X1, X2, degree=2, c=1.0):
    return (np.dot(X1, X2.T) + c) ** degree


def fit_kernel_svm(X, y, c=1.0, degree=2):
    n_samples, n_features = X.shape
    K = polynomial_kernel(X, X, degree=degree)
    alpha = cp.Variable(n_samples)
    P = np.outer(y, y) * K
    P = P + np.eye(n_samples) * 1e-05
    objective = cp.Minimize(0.5 * cp.quad_form(alpha, P) - cp.sum(alpha))
    constraints = [alpha >= 0, alpha <= c, cp.sum(cp.multiply(alpha, y)) == 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    support = np.where(alpha.value > 1e-05)[0]
    sv = support[0]
    b = y[sv] - np.sum(alpha.value * y * K[:, sv])
    return (alpha.value, b)


def predict_kernel_svm(X_train, y_train, alpha, b, X_test, degree=2, c=1.0):
    K = polynomial_kernel(X_train, X_test, degree=degree)
    decision = np.sum((alpha * y_train).reshape(-1, 1) * K, axis=0)
    return np.sign(decision + b)


def plot_kernel_svm(X, y, alpha, b, degree=2):
    plt.figure(figsize=(8, 6))
    x_min, x_max = (X[:, 0].min() - 1, X[:, 0].max() + 1)
    y_min, y_max = (X[:, 1].min() - 1, X[:, 1].max() + 1)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_kernel_svm(X, y, alpha, b, grid_points, degree=degree)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
    support = np.where(alpha > 1e-05)[0]
    plt.scatter(X[support, 0], X[support, 1], s=120, facecolors="none", edgecolors="k")
    plt.title("Kernel SVM with Polynomial Kernel")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.savefig("../figures/nonlinear_kernel_svm.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    X, y = generate_nonlinear_data()
    alpha, b = fit_kernel_svm(X, y, degree=2, c=1.0)
    plot_kernel_svm(X, y, alpha, b, degree=2)
