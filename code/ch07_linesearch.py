"""Chapter 7: Armijo backtracking, Wolfe conditions, and BB steps."""
import numpy as np
from students import X, y, m, H, w_star, loss, grad

Lstar = loss(w_star)
lam = np.linalg.eigvalsh(H)
mu, be = lam[0], lam[-1]


def backtrack(L, g, w, d, eta0, tau, c1, Lw=None):
    """Armijo backtracking along d; returns (eta, list of (eta, phi))."""
    Lw = L(w) if Lw is None else Lw
    slope = g @ d
    eta, trials = eta0, []
    while True:
        phi = L(w + eta * d)
        trials.append((eta, phi))
        if phi <= Lw + c1 * eta * slope:
            return eta, trials
        eta *= tau


# --- Example: a two-weight quadratic ------------------------------------
print("toy: L = (w1^2 + 10 w2^2)/2 at w = (2, 1)")
Ht = np.diag([1.0, 10.0])
Lt = lambda w: 0.5 * w @ Ht @ w
gt = lambda w: Ht @ w
w = np.array([2.0, 1.0])
g = gt(w)
print(f"  L = {Lt(w):g}, g = {g}, g'g = {g @ g:g}, g'Hg = {g @ Ht @ g:g}")
eta, tr = backtrack(Lt, g, w, -g, 1.0, 0.5, 0.5)
for e, p in tr:
    print(f"  eta = {e:.4f}  phi = {p:9.4f}  line = {Lt(w) - 0.5 * e * g @ g:8.4f}")
es = g @ g / (g @ Ht @ g)
print(f"  accepted {eta}, bound 2tau(1-c1)/beta = {2 * 0.5 * 0.5 / 10:.4f},"
      f" exact = {es:.4f}, w1 = {w - eta * g}")
for c1, c2 in ((0.1, 0.9),):
    print(f"  c1={c1}, c2={c2}: Armijo eta <= {2 * (1 - c1) * es:.4f},"
          f" curvature eta >= {(1 - c2) * es:.5f},"
          f" strong eta <= {(1 + c2) * es:.4f}")

# --- Students: first backtracking step ----------------------------------
w0 = np.zeros(3)
g0 = grad(w0)
gg, gHg, L0 = g0 @ g0, g0 @ H @ g0, loss(w0)
print(f"\nstudents: L0 = {L0:.4f}, g'g = {gg:.3f}, g'Hg = {gHg:.2f},"
      f" exact step = {gg / gHg:.5f}, 2/beta = {2 / be:.5f}")
print(f"  mu = {mu:.4f}, beta = {be:.4f}, kappa = {be / mu:.2f}")
c1 = 1e-4
eta, tr = backtrack(loss, g0, w0, -g0, 1.0, 0.5, c1)
for e, p in tr:
    print(f"  eta = {e:.6f}  phi = {p:13.3f}  line = {L0 - c1 * e * gg:10.3f}")
print(f"  bound 2tau(1-c1)/beta = {2 * 0.5 * (1 - c1) / be:.5f}")
dphi = -gg + eta * gHg
print(f"  phi'(0) = {-gg:.3f}, phi'({eta}) = {dphi:.3f}, "
      f"0.9 phi'(0) = {-0.9 * gg:.3f}")
es = gg / gHg
print(f"  first-line intervals, c1=1e-4, c2=0.9: curvature eta >= "
      f"{0.1 * es:.5f}, Armijo eta <= {2 * (1 - c1) * es:.5f}, "
      f"strong eta <= {1.9 * es:.5f}")
w1 = w0 - eta * g0
s0, z0 = w1 - w0, grad(w1) - g0
print(f"  w1 = {w1}, L(w1) = {loss(w1):.3f}")
print(f"  BB: s0's0 = {s0 @ s0:.4f}, s0'z0 = {s0 @ z0:.4f}, "
      f"eta1 = {s0 @ s0 / (s0 @ z0):.5f}")


# --- Students: runs to the target ---------------------------------------
target = Lstar + 0.01


def run(rule, maxit=10**6):
    """rule(w, g, Lw, prev) -> (eta, loss evals); returns counts."""
    w = np.zeros(3)
    Lw, nL, nG, prev = loss(w), 1, 0, None
    etas = []
    while Lw > target and len(etas) < maxit:
        g = grad(w); nG += 1
        eta, k = rule(w, g, Lw, prev)
        nL += k
        prev = (w.copy(), g.copy(), eta)
        w = w - eta * g
        Lw = loss(w)
        etas.append(eta)
    return len(etas), nG, nL, np.array(etas)


def fixed(e):
    return lambda w, g, Lw, prev: (e, 1)


def exact(w, g, Lw, prev):
    return g @ g / (g @ H @ g), 1


def bt_cold(w, g, Lw, prev):
    eta, tr = backtrack(loss, g, w, -g, 1.0, 0.5, 1e-4, Lw)
    return eta, len(tr)


def bt_warm(w, g, Lw, prev):
    e0 = 1.0 if prev is None else 2 * prev[2]
    eta, tr = backtrack(loss, g, w, -g, e0, 0.5, 1e-4, Lw)
    return eta, len(tr)


def bb(w, g, Lw, prev):
    if prev is None:
        return bt_cold(w, g, Lw, prev)
    s = w - prev[0]
    z = g - prev[1]
    return s @ s / (s @ z), 1


print(f"\nruns to L <= L* + 0.01 = {target:.4f} from w = 0")
print("rule            steps  grads  losses  min eta  max eta")
res = {}
for name, rule in (("fixed 0.03", fixed(0.03)), ("exact", exact),
                   ("backtrack cold", bt_cold), ("backtrack warm", bt_warm),
                   ("BB", bb)):
    n, nG, nL, etas = run(rule)
    res[name] = etas
    print(f"{name:15s} {n:6d} {nG:6d} {nL:7d} {etas.min():8.5f} {etas.max():8.5f}")
e = res["backtrack cold"]
vals, cnt = np.unique(e, return_counts=True)
print("cold accepted steps:", {f"{v:.6f}": int(c) for v, c in zip(vals, cnt)})
w = np.zeros(3)
bl = [loss(w)]
prev = None
for t in range(11):
    g = grad(w)
    eta, _ = bb(w, g, bl[-1], prev)
    print(f"  BB step {t + 1}: eta = {eta:.5f}")
    prev = (w.copy(), g.copy(), eta)
    w = w - eta * g
    bl.append(loss(w))
print(f"1/mu = {1 / mu:.2f}, 1/lam2 = {1 / lam[1]:.4f}")
print("BB losses:", " ".join(f"{v:.4g}" for v in bl))
e = res["backtrack warm"]
print(f"warm: share of steps with eta > 2/beta: {np.mean(e > 2 / be):.3f}")
print("warm first 8 etas:", np.round(e[:8], 5))

# --- Students: penalised cross-entropy ----------------------------------
t_lab = (y >= 65).astype(float)
rho = 0.01
P = np.diag([0.0, 1.0, 1.0])
sig = lambda z: 1 / (1 + np.exp(-z))


def ce(w):
    z = X @ w
    return np.mean(np.logaddexp(0, z) - t_lab * z) + 0.5 * rho * w[1:] @ w[1:]


def ce_g(w):
    return X.T @ (sig(X @ w) - t_lab) / m + rho * P @ w


def ce_H(w):
    p = sig(X @ w)
    return X.T @ np.diag(p * (1 - p)) @ X / m + rho * P


w = np.zeros(3)
for _ in range(40):
    w = w - np.linalg.solve(ce_H(w), ce_g(w))
ws_ce, Ls_ce = w, ce(w)
beta_ce = be / 4 + rho
print(f"\nCE: L* = {Ls_ce:.6f}, beta bound = {beta_ce:.4f}, "
      f"1/beta = {1 / beta_ce:.5f}, lam_max H(w*) = "
      f"{np.linalg.eigvalsh(ce_H(ws_ce))[-1]:.4f}")


def ce_run(kind, tol, maxit=10**6):
    w = np.zeros(3)
    Lw, nL, n, etas = ce(w), 1, 0, []
    while Lw - Ls_ce > tol and n < maxit:
        g = ce_g(w)
        if kind == "fixed":
            eta, k = 1 / beta_ce, 1
        else:
            e0 = 1.0 if not etas else 2 * etas[-1]
            eta, tr = backtrack(ce, g, w, -g, e0, 0.5, 1e-4, Lw)
            k = len(tr)
        w = w - eta * g
        Lw = ce(w); nL += k; n += 1
        etas.append(eta)
    return n, nL, np.array(etas)


print("tol     fixed-steps  bt-steps  bt-losses  bt median eta  bt max eta")
for tol in (1e-2, 1e-4, 1e-6):
    nf, _, _ = ce_run("fixed", tol)
    nb, nl, et = ce_run("bt", tol)
    print(f"{tol:.0e} {nf:12d} {nb:9d} {nl:10d} {np.median(et):14.4f} "
          f"{et.max():11.4f}")

# --- answers to one-star exercises --------------------------------------
# ex 7.1: phi(eta) = 4 - 8 eta + 8 eta^2 (L = w^2 at w = 2... see text)
f = lambda e: (2 - 4 * e) ** 2
for e in (1.0, 0.5, 0.25):
    print(f"ex 7.1: eta = {e}: phi = {f(e):g}, line = {4 - 0.5 * e * 16:g}")
es = 1 / 2
print(f"ex 7.2: c1=0.2 Armijo eta <= {2 * 0.8 * es:.3f},"
      f" c2=0.5 curvature eta >= {0.5 * es:.3f}, strong eta <= {1.5 * es:.3f}")
print(f"ex 7.3: steps bound min(1, 2*0.5*0.9/50) = {min(1, 2 * 0.5 * 0.9 / 50):.3f},"
      f" trials <= {1 + int(np.ceil(np.log(50 / 1.8) / np.log(2)))}")
