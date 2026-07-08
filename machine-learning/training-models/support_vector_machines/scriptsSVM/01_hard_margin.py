import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def generate_linearly_seperable_data():
    X, y = make_blobs(
        n_samples=50, centers=2, random_state=42, cluster_std=1.0, n_features=2
    )
    y = np.where(y == 0, -1, 1)
    return (X, y)


def fit_hard_margin_svm(X, y):
    n_samples, n_features = X.shape
    w = cp.Variable(n_features)
    b = cp.Variable()
    objective = cp.Minimize(0.5 * cp.sum_squares(w))
    constraints = [y[i] * (w @ X[i] + b) >= 1 for i in range(n_samples)]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return (w.value, b.value)


def plot_decision_boundary(X, y, w, b):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", alpha=0.7, edgecolor="black")
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 50)
    yy = np.linspace(ylim[0], ylim[1], 50)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = (xy @ w + b).reshape(XX.shape)
    ax.contour(
        XX,
        YY,
        Z,
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
    )
    margins = y * (X @ w + b)
    support_vectors = X[np.isclose(margins, 1.0, atol=0.0001)]
    ax.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=150,
        linewidth=1.5,
        facecolors="none",
        edgecolors="k",
        label="Support Vectors",
    )
    plt.title("Hard Margin SVM (cvxpy)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.show()


if __name__ == "__main__":
    X, y = generate_linearly_seperable_data()
    w_opt, b_opt = fit_hard_margin_svm(X, y)
    print(f"optimize edilmis agirliklar (w): {w_opt}")
    print(f"optimize edilmis bias (b): {b_opt}")
    plot_decision_boundary(X, y, w_opt, b_opt)
