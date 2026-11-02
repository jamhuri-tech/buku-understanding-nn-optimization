"""Chapter 0: numbers for the toolkit section on the eight students."""
import numpy as np

x1 = np.array([1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.5])
x2 = np.array([5.0, 8.0, 6.0, 7.0, 5.5, 8.0, 6.5, 7.5])
y = np.array([48, 66, 60, 68, 64, 80, 74, 86.0])
m = len(y)
X = np.column_stack([np.ones(m), x1, x2])

def loss(w):
    r = X @ w - y
    return r @ r / (2 * m)

def grad(w):
    return X.T @ (X @ w - y) / m

H = X.T @ X / m
w = np.zeros(3)
print("mean x1, x2, y:", x1.mean(), x2.mean(), y.mean())
print("L(0) =", round(loss(w), 3))
print("g(0) =", grad(w).round(3))
print("H =\n", H.round(4))
lam, Q = np.linalg.eigh(H)
print("eigenvalues:", lam.round(4))
print("condition number:", round(lam[-1] / lam[0], 1))
ws = np.linalg.solve(H, X.T @ y / m)
print("w* =", ws.round(3), " L(w*) =", round(loss(ws), 3))
for eta in (0.01, 0.03, 0.04):
    w1 = w - eta * grad(w)
    print(f"eta={eta}: w1 = {w1.round(3)}, L(w1) = {loss(w1):.3f}")
print("2/lambda_max =", round(2 / lam[-1], 4))
