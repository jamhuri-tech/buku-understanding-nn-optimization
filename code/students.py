"""The eight students used throughout the book."""
import numpy as np

x1 = np.array([1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.5])   # study, h/day
x2 = np.array([5.0, 8.0, 6.0, 7.0, 5.5, 8.0, 6.5, 7.5])   # sleep, h/day
y = np.array([48, 66, 60, 68, 64, 80, 74, 86.0])          # exam score
m = len(y)
X = np.column_stack([np.ones(m), x1, x2])                 # X tilde


def loss(w):
    r = X @ w - y
    return r @ r / (2 * m)


def grad(w):
    return X.T @ (X @ w - y) / m


H = X.T @ X / m
w_star = np.linalg.solve(H, X.T @ y / m)


def gd(eta, steps, w=None):
    """Gradient descent from w (default 0); returns the list of iterates."""
    w = np.zeros(3) if w is None else w.copy()
    path = [w.copy()]
    for _ in range(steps):
        w = w - eta * grad(w)
        path.append(w.copy())
    return path
