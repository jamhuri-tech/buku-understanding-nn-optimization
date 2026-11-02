"""Chapter 4: Newton's method on one weight and on the students."""
import numpy as np
from students import X, y, m, H, w_star, loss, grad

np.set_printoptions(precision=4, suppress=True)

# --- one weight: f(w) = ln(1 + e^w) - 0.6 w from Chapter 0 -----------
sig = lambda z: 1 / (1 + np.exp(-z))
f1 = lambda w: np.log1p(np.exp(w)) - 0.6 * w
d1 = lambda w: sig(w) - 0.6
d2 = lambda w: sig(w) * (1 - sig(w))
wmin = np.log(1.5)
print(f"one weight: minimiser ln 1.5 = {wmin:.10f}")
for w0 in (1.0, 2.0, 2.5):
    w = w0
    print(f"start w0 = {w0}")
    for t in range(6):
        print(f"  t={t}  w={w: .10f}  f'={d1(w): .3e}  f''={d2(w):.4f}"
              f"  error={w - wmin: .3e}")
        if abs(w) > 50:
            break
        w = w - d1(w) / d2(w)

# --- worked quadratic --------------------------------------------------
Hq = np.array([[4.0, 2.0], [2.0, 3.0]])
bq = np.array([6.0, 5.0])
print("quadratic: H^-1 b =", np.linalg.solve(Hq, bq),
      " eig(H) =", np.linalg.eigvalsh(Hq))
eta = 2 / sum(np.linalg.eigvalsh(Hq)[[0, 1]])
w = np.zeros(2)
for t in range(1, 4):
    w = w - eta * (Hq @ w - bq)
    print(f"  GD eta={eta:.4f} step {t}: w = {w}")

# --- indefinite Hessian and Levenberg ------------------------------------
Hi = np.diag([2.0, -1.0]); gi = np.array([1.0, 1.0])
for nu in (0.0, 2.0, 10.0):
    s = -np.linalg.solve(Hi + nu * np.eye(2), gi)
    print(f"levenberg nu={nu:4.1f}: step = {s}, g.step = {gi @ s:.4f}")

# --- students, squared error: one Newton step ----------------------------
w0 = np.zeros(3)
g0 = grad(w0)
w1 = w0 - np.linalg.solve(H, g0)
print("MSE: g0 =", g0, " Newton w1 =", np.round(w1, 3),
      f" L(w1) = {loss(w1):.4f}, |w1 - w*| = {np.linalg.norm(w1 - w_star):.1e}")
print("row checks of H w* = X^T y / m:", H @ w_star, X.T @ y / m)
lam = np.linalg.eigvalsh(H)
print(f"eig(H) = {lam}, kappa = {lam[-1] / lam[0]:.2f}")
for name, A in (("centred", np.column_stack([np.ones(m), X[:, 1] - X[:, 1].mean(),
                                               X[:, 2] - X[:, 2].mean()])),
                ("standard", np.column_stack([np.ones(m),
                                              (X[:, 1] - X[:, 1].mean()) / X[:, 1].std(),
                                              (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()]))):
    Ha = A.T @ A / m
    v1 = -np.linalg.solve(Ha, A.T @ (A @ np.zeros(3) - y) / m)
    r = A @ v1 - y
    print(f"  {name}: Newton v1 = {v1}, L = {r @ r / (2 * m):.4f}")

# --- students, penalised cross-entropy ----------------------------------
t_lab = (y >= 65).astype(float)
rho = 0.01
P = np.diag([0.0, 1.0, 1.0])


def ce(w):
    z = X @ w
    return np.mean(np.logaddexp(0, z) - t_lab * z) + 0.5 * rho * w[1:] @ w[1:]


def ce_g(w):
    return X.T @ (sig(X @ w) - t_lab) / m + rho * P @ w


def ce_H(w):
    p = sig(X @ w)
    return X.T @ np.diag(p * (1 - p)) @ X / m + rho * P


w = np.zeros(3)
path = [w.copy()]
for _ in range(30):
    w = w - np.linalg.solve(ce_H(w), ce_g(w))
    path.append(w.copy())
ws = path[-1]
Ls = ce(ws)
print("CE: w* =", np.round(ws, 3), f" L* = {Ls:.6f}")
print("CE first Newton step from 0: H(0) =\n", ce_H(np.zeros(3)))
print("  g(0) =", ce_g(np.zeros(3)), " w1 =", path[1])
print(" t       w0        w1       w2      L - L*      |g|      |e|    "
      "|e+|/|e|^2")
for t in range(10):
    e = np.linalg.norm(path[t] - ws)
    e1 = np.linalg.norm(path[t + 1] - ws)
    ratio = f"{e1 / e**2:8.3f}" if e1 > 1e-12 else "      --"
    print(f"{t:2d} {path[t][0]:9.4f} {path[t][1]:8.4f} {path[t][2]:8.4f}"
          f" {ce(path[t]) - Ls:10.3e} {np.linalg.norm(ce_g(path[t])):9.2e}"
          f" {e:9.2e} {ratio}")
    if e < 1e-10:
        break
lH = np.linalg.eigvalsh(ce_H(ws))
lH0 = np.linalg.eigvalsh(ce_H(np.zeros(3)))
beta = np.linalg.eigvalsh(H)[-1] / 4 + rho
print(f"eig H(w*) = {lH}, eig H(0) = {lH0}, beta bound = {beta:.4f}")

# gradient descent on the same loss with eta = 1/beta
w = np.zeros(3)
gaps_gd = [ce(w) - Ls]
for k in range(1, 200001):
    w = w - ce_g(w) / beta
    gaps_gd.append(ce(w) - Ls)
gaps_gd = np.array(gaps_gd)
for tol in (1e-2, 1e-4, 1e-6):
    hit = np.argmax(gaps_gd <= tol) if (gaps_gd <= tol).any() else None
    print(f"GD eta=1/beta: first step with gap <= {tol:.0e}: {hit}")
nt = next(t for t in range(len(path)) if ce(path[t]) - Ls <= 1e-6)
print(f"Newton: first step with gap <= 1e-06: {nt}")
print(f"GD gap after 1000, 10000, 100000 steps: {gaps_gd[1000]:.3e}, "
      f"{gaps_gd[10000]:.3e}, {gaps_gd[100000]:.3e}")

# --- cost table ----------------------------------------------------------
print("cost: n, grad bytes, Hessian bytes (float32), n^3/3 flops, "
      "time at 1e14 flop/s")
for n in (1e3, 1e6, 1e9):
    print(f"  {n:.0e} {4 * n:.0e} {4 * n * n:.0e} {n**3 / 3:.1e} "
          f"{n**3 / 3 / 1e14:.1e} s = {n**3 / 3 / 1e14 / 3.156e7:.1e} years")

# --- answers to one-star exercises --------------------------------------
Ha = np.array([[2.0, 1.0], [1.0, 2.0]]); ga = np.array([3.0, 0.0])
print("ex 4.1: step =", -np.linalg.solve(Ha, ga))
w = 0.0
for t in range(1, 4):
    w = w - (2 * w + np.exp(w)) / (2 + np.exp(w))
    print(f"ex 4.2: w{t} = {w:.6f}")
n = 1e7
print(f"ex 4.3: Hessian bytes = {4 * n * n:.1e}, solve = {n**3 / 3 / 1e14:.2e} s"
      f" = {n**3 / 3 / 1e14 / 86400:.1f} days")
w1r = np.round(w_star, 3)
print("example 4.4 rows:", [np.round(H[i] * w1r, 3) for i in range(3)],
      np.round(H @ w1r, 3))
print(f"example 4.1: sigma(1) = {sig(1.0):.4f}, f'''(w*)/(2f''(w*)) ="
      f" {d2(wmin) * (1 - 2 * sig(wmin)) / (2 * d2(wmin)):.4f}")
