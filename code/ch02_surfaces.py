"""Chapter 2: shapes of the students' losses."""
import numpy as np
from students import X, y, m, H, w_star, loss, x1, x2

lam, Q = np.linalg.eigh(H)
print("MSE: eigenvalues", lam.round(4), "kappa", round(lam[-1] / lam[0], 1))
print("q1 =", Q[:, 0].round(3), " q3 =", Q[:, 2].round(3))

# centred features: subtract the means
Xc = np.column_stack([np.ones(m), x1 - x1.mean(), x2 - x2.mean()])
Hc = Xc.T @ Xc / m
lc = np.linalg.eigvalsh(Hc)
print("centred: H =\n", Hc.round(4), "\n eigenvalues", lc.round(4),
      "kappa", round(lc[-1] / lc[0], 1))
# standardised features
s1, s2 = x1.std(), x2.std()
Xs = np.column_stack([np.ones(m), (x1 - x1.mean()) / s1, (x2 - x2.mean()) / s2])
Hs = Xs.T @ Xs / m
ls = np.linalg.eigvalsh(Hs)
print("std devs:", round(s1, 4), round(s2, 4))
print("standardised: H =\n", Hs.round(4), "\n eigenvalues", ls.round(4),
      "kappa", round(ls[-1] / ls[0], 2))

# pass/fail and logistic loss
t = (y >= 65).astype(float)
print("pass labels:", t.astype(int))
sig = lambda z: 1 / (1 + np.exp(-z))

def ce(w, lam2=0.0):
    z = X @ w
    # -log p = log(1 + e^{-z}), -log(1 - p) = log(1 + e^{z})
    return (np.mean(t * np.logaddexp(0, -z) + (1 - t) * np.logaddexp(0, z))
            + 0.5 * lam2 * w[1:] @ w[1:])

def ce_grad(w, lam2=0.0):
    p = sig(X @ w)
    g = X.T @ (p - t) / m
    g[1:] += lam2 * w[1:]
    return g

def ce_hess(w, lam2=0.0):
    p = sig(X @ w)
    D = p * (1 - p)
    Hh = X.T @ (D[:, None] * X) / m
    Hh[1:, 1:] += lam2 * np.eye(2)
    return Hh

w0 = np.zeros(3)
print("CE(0) =", round(ce(w0), 4), " grad(0) =", ce_grad(w0).round(4))
print("H_ce(0) = H/4:", np.allclose(ce_hess(w0), H / 4))
# separating line? try w along a direction
for c in (1, 2, 5, 10):
    wd = c * np.array([-13.0, 1.0, 1.5])
    print(f"scale {c}: margins min = {np.min((2*t-1)*(X@wd)):.2f}, CE = {ce(wd):.6f}, CE_reg = {ce(wd, 0.01):.4f}")
# Newton on regularised CE
for lam2 in (0.01,):
    w = np.zeros(3)
    for _ in range(50):
        w = w - np.linalg.solve(ce_hess(w, lam2), ce_grad(w, lam2))
    ev = np.linalg.eigvalsh(ce_hess(w, lam2))
    print(f"lambda={lam2}: w* = {w.round(3)}, CE = {ce(w, lam2):.4f},"
          f" eig = {ev.round(5)}, kappa = {ev[-1]/ev[0]:.1f}")
    print("eig at 0:", np.linalg.eigvalsh(ce_hess(np.zeros(3), lam2)).round(4))
