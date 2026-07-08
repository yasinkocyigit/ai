import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def generate_data():
    X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.0)
    y = np.where(y == 0, -1, 1)
    return (X, y)


def linear_kernel(x1, x2):
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    return np.dot(x1, x2.T)


class SequentialMinimalOptimization:

    def __init__(self, C=1.0, tol=0.001, max_passes=200):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes

    def fit(self, X, y):
        print("C =", self.C)
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.alpha = np.zeros(self.n_samples)
        self.b = 0.0
        self.K = linear_kernel(self.X, self.X)
        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(self.n_samples):
                E_i = self.compute_error(i)
                if self.violates_kkt(i, E_i):
                    j = self.select_second_alpha(i, E_i)
                    E_j = self.compute_error(j)
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    L, H = self.compute_bounds(i, j)
                    if L == H:
                        continue
                    eta = self.compute_eta(i, j)
                    if eta >= 0:
                        continue
                    self.alpha[j] -= self.y[j] * (E_i - E_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    if abs(self.alpha[j] - alpha_j_old) < 1e-05:
                        continue
                    self.alpha[i] += (
                        self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])
                    )
                    self.alpha[i] = np.clip(self.alpha[i], 0, self.C)
                    self.update_bias(i, j, alpha_i_old, alpha_j_old, E_i, E_j)
                    num_changed_alphas += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        decision = self.decision_function(X)
        print("y*f(x) min/max:", np.min(y * decision), np.max(y * decision))
        return self

    def compute_error(self, index):
        return self.decision_function(self.X[index])[0] - self.y[index]

    def violates_kkt(self, i, E_i):
        return (
            self.y[i] * E_i < -self.tol
            and self.alpha[i] < self.C
            or (self.y[i] * E_i > self.tol and self.alpha[i] > 0)
        )

    def compute_bounds(self, i, j):
        if self.y[i] != self.y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        return (L, H)

    def compute_eta(self, i, j):
        return 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]

    def update_bias(self, i, j, alpha_i_old, alpha_j_old, E_i, E_j):
        b1 = (
            self.b
            - E_i
            - self.y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i]
            - self.y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j]
        )
        b2 = (
            self.b
            - E_j
            - self.y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j]
            - self.y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j]
        )
        if 0 < self.alpha[i] < self.C:
            self.b = b1
        elif 0 < self.alpha[j] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

    def select_second_alpha(self, i, E_i):
        j = i
        while j == i:
            j = np.random.randint(0, self.n_samples)
        return j

    def decision_function(self, X):
        X = np.atleast_2d(X)
        K = linear_kernel(self.X, X)
        decision = np.sum((self.alpha * self.y).reshape(-1, 1) * K, axis=0)
        decision += self.b
        return decision

    def predict(self, X):
        return np.sign(self.decision_function(X))


def plot_data(model):
    plt.figure(figsize=(8, 6))
    X = model.X
    y = model.y
    x_min, x_max = (X[:, 0].min() - 1, X[:, 0].max() + 1)
    y_min, y_max = (X[:, 1].min() - 1, X[:, 1].max() + 1)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    decision = model.decision_function(grid_points)
    decision = decision.reshape(xx.shape)
    plt.contourf(xx, yy, np.sign(decision), alpha=0.3, cmap="bwr")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
    plt.contour(
        xx,
        yy,
        decision,
        levels=[-1, 0, 1],
        colors=["black", "green", "black"],
        linestyles=["--", "-", "--"],
        linewidths=2,
    )
    support = model.alpha > 1e-05
    plt.scatter(
        X[support, 0],
        X[support, 1],
        s=120,
        facecolors="none",
        edgecolors="k",
        linewidth=2,
    )
    plt.title("Training Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    X, y = generate_data()
    print("X Shape:", X.shape)
    print("Y Shape:", y.shape)
    model = SequentialMinimalOptimization(C=1.0, tol=0.001, max_passes=200)
    model.fit(X, y)
    plot_data(model)
    print("\n\n\n")
    pred = model.predict(X)
    print("Accuracy:", np.mean(pred == y))
    print()
    print("Support Vector Sayısı:", np.sum(model.alpha > 1e-05))
    print()
    print("Alpha:")
    print(model.alpha)
    w = np.sum((model.alpha * y).reshape(-1, 1) * X, axis=0)
    print()
    print("w =", w)
    print("b =", model.b)
    print("margin =", 2 / np.linalg.norm(w))
    support = model.alpha > 1e-05
    for idx in np.where(support)[0]:
        print(
            idx,
            model.alpha[idx],
            model.decision_function(model.X[idx])[0],
            model.y[idx],
        )
