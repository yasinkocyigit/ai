import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def generate_overlapping_data():
    X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=2.5)
    y = np.where(y <= 0, -1, 1)
    return (X, y)


def fit_soft_margin_svm(X, y, C):
    n_samples, n_features = X.shape
    w = cp.Variable(n_features)
    b = cp.Variable()
    xi = cp.Variable(n_samples)
    objective = cp.Minimize(0.5 * cp.sum_squares(w) + C * cp.sum(xi))
    constraints = [cp.multiply(y, X @ w + b) >= 1 - xi, xi >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return (w.value, b.value)


def plot_soft_margin_comparison(X, y, C_values):
    fig, axes = plt.subplots(1, len(C_values), figsize=(14, 6))
    for ax, C in zip(axes, C_values):
        w, b = fit_soft_margin_svm(X, y, C)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", alpha=0.7, edgecolors="k")
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
        support_vectors = X[margins <= 1.0 + 0.0001]
        ax.scatter(
            support_vectors[:, 0],
            support_vectors[:, 1],
            s=100,
            linewidth=1.5,
            facecolors="none",
            edgecolors="k",
        )
        ax.set_title(f"Soft Margin SVM (c = {C})")
        ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y = generate_overlapping_data()
    C_values = [0.05, 5.0, 10.0]
    plot_soft_margin_comparison(X, y, C_values)
