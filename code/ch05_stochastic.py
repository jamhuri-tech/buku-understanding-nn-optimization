"""Chapter 5: stochastic approximation, LMS and mini-batches."""
import numpy as np
from students import y, m, x1, x2

np.set_printoptions(precision=4, suppress=True)

# Centred features (Chapter 2): kappa = 2.83 instead of 2430.
Xc = np.column_stack([np.ones(m), x1 - x1.mean(), x2 - x2.mean()])
Hc = Xc.T @ Xc / m
wc = np.linalg.solve(Hc, Xc.T @ y / m)
lam = np.linalg.eigvalsh(Hc)
mu, beta = lam[0], lam[-1]


def L(w):
    r = Xc @ w - y
    return r @ r / (2 * m)


Lstar = L(wc)


def per_sample(w, A=Xc):
    """Rows are the per-sample gradients g_i = (a_i'w - y_i) a_i."""
    return (A @ w - y)[:, None] * A


def spread(w, A=Xc):
    """Variance of g_i around its mean, as in Equation (0.expect)."""
    G = per_sample(w, A)
    return np.mean(np.sum((G - G.mean(0)) ** 2, axis=1))


def sgd(eta, epochs, k=1, seed=0, replace=True):
    """Mini-batch SGD from 0; eta(t) gives the step at iteration t.
    Returns the gap L - L* after each epoch (entry 0 is the start)."""
    rng = np.random.default_rng(seed)
    w, t, gaps = np.zeros(3), 0, [L(np.zeros(3)) - Lstar]
    for _ in range(epochs):
        order = (rng.integers(0, m, m) if replace
                 else rng.permutation(m))
        for j in range(0, m, k):
            B = order[j:j + k]
            g = Xc[B].T @ (Xc[B] @ w - y[B]) / len(B)
            w = w - eta(t) * g
            t += 1
        gaps.append(L(w) - Lstar)
    return np.array(gaps)


def gd(eta, epochs):
    w, gaps = np.zeros(3), [L(np.zeros(3)) - Lstar]
    for _ in range(epochs):
        w = w - eta * Xc.T @ (Xc @ w - y) / m
        gaps.append(L(w) - Lstar)
    return np.array(gaps)


C_DEC, T0 = 2 / mu, 26.0
dec = lambda t: C_DEC / (t + T0)


def mean_run(eta, T, seed=1):
    """Robbins-Monro for the mean of N(5, 2^2) readings."""
    z = np.random.default_rng(seed).normal(5.0, 2.0, T)
    w, path = 0.0, []
    for t in range(1, T + 1):
        w = w - eta(t) * (w - z[t - 1])
        path.append(w)
    return np.array(path)


if __name__ == "__main__":
    # --- Robbins-Monro: running mean -------------------------------------
    z = [4.0, 7.0, 5.0, 8.0]
    w = 0.0
    for t, zt in enumerate(z, 1):
        w = w - (1 / t) * (w - zt)
        print(f"RM eta=1/t  t={t}  w={w:.4f}")
    w = 0.0
    for t, zt in enumerate(z, 1):
        w = w - 0.5 * (w - zt)
        print(f"RM eta=0.5  t={t}  w={w:.4f}")
    for eta in (0.5, 0.1, 0.01):
        print(f"floor eta={eta}: variance eta*s2/(2-eta) = "
              f"{eta * 4 / (2 - eta):.4f}")
    T = 2000
    for name, e in (("0.1", lambda t: 0.1), ("1/t", lambda t: 1 / t)):
        p = mean_run(e, T)
        print(f"mean run eta={name}: w_T={p[-1]:.4f}  "
              f"spread of last 1000 = {p[-1000:].std():.4f}")

    # --- Kiefer-Wolfowitz on f(w) = w^4/4 --------------------------------
    f = lambda w: w ** 4 / 4
    print(f"KW f(1.5)={f(1.5):.4f} f(0.5)={f(0.5):.4f} "
          f"exact difference={(f(1.5) - f(0.5)) / 1.0:.4f}")
    est = (1.30 - 0.03) / (2 * 0.5)
    print(f"KW noisy estimate={est:.4f}  step to {1 - 0.2 * est:.4f}")
    print(f"KW sd of estimate sigma/(sqrt2 c)={0.1 / np.sqrt(2) / 0.5:.4f}"
          f"  best c={(0.1 / 2) ** (1 / 3):.4f}")

    # --- the centred students ----------------------------------------------
    print("centred w* =", wc, f" L* = {Lstar:.4f}")
    print(f"mu={mu:.4f} beta={beta:.4f} kappa={beta / mu:.4f}")
    nx = np.sum(Xc ** 2, axis=1)
    print("|x_i|^2 =", nx, " 2/max =", f"{2 / nx.max():.4f}")

    # one LMS pass by hand, eta = 0.05, samples 1..3
    w = np.zeros(3)
    for i in range(3):
        r = Xc[i] @ w - y[i]
        w = w - 0.05 * r * Xc[i]
        print(f"LMS sample {i + 1}: r={r:.4f}  w={w}  L={L(w):.4f}")
    w = np.zeros(3)
    for i in range(m):
        w = w - 0.05 * (Xc[i] @ w - y[i]) * Xc[i]
    print(f"LMS after one ordered epoch: w={w}  L={L(w):.4f}")

    # per-sample gradients and their variance
    for name, w in (("w=0", np.zeros(3)), ("w*", wc)):
        G = per_sample(w)
        print(f"{name}: mean g = {G.mean(0)}")
        print(f"{name}: |g_i|^2 =", np.sum(G ** 2, axis=1))
        print(f"{name}: variance = {spread(w):.4f}  |grad L|^2 = "
              f"{G.mean(0) @ G.mean(0):.2f}")
    print("residuals at w* =", Xc @ wc - y)
    print("g_1 at w* =", per_sample(wc)[0], " g_2 =", per_sample(wc)[1])
    print("batch {1,2} at w* =", per_sample(wc)[:2].mean(0))
    from students import X
    print(f"raw features: variance at 0 = {spread(np.zeros(3), X):.1f}"
          f"  max |x_i|^2 = {np.max(np.sum(X ** 2, axis=1)):.2f}")

    # mini-batch variance at w*
    s2 = spread(wc)
    for k in (1, 2, 4, 8):
        print(f"k={k}: with repl {s2 / k:8.4f}   without "
              f"{s2 / k * (m - k) / (m - 1):8.4f}")

    # GD against LMS, gaps averaged over 20 seeds
    E, S = 1000, 20
    print(f"decreasing step: c={C_DEC:.4f} t0={T0:.0f} "
          f"eta_0={dec(0):.4f}")
    g0 = gd(0.4, E)
    print("GD eta=0.4: first epoch with gap < 1e-6:",
          int(np.argmax(g0 < 1e-6)))
    avg = lambda fn: np.mean([fn(s) for s in range(S)], axis=0)
    runs = {}
    for eta in (0.1, 0.05, 0.02):
        runs[f"LMS eta={eta}"] = avg(lambda s, e=eta:
                                     sgd(lambda t: e, E, seed=s))
    runs["LMS decreasing"] = avg(lambda s: sgd(dec, E, seed=s))
    for name, g in runs.items():
        print(f"{name:15s} e10={g[10]:.2e} e100={g[100]:.2e} "
              f"e1000={g[1000]:.2e} floor={g[501:].mean():.2e}")
    for eta in (0.1, 0.05, 0.02):
        fl = runs[f"LMS eta={eta}"][501:].mean()
        print(f"floor/eta for eta={eta}: {fl / eta:.3f}  "
              f"eta*s2/4 = {eta * s2 / 4:.2e}")
    g = runs["LMS decreasing"]
    print("decreasing: epoch*gap at 100, 1000 =",
          f"{100 * g[100]:.2f} {1000 * g[1000]:.2f}")

    # mini-batch sizes, eta = 0.1, draws with and without replacement
    for k in (1, 2, 4, 8):
        fl = []
        for rep in (True, False):
            g = avg(lambda s: sgd(lambda t: 0.1, E, k=k, seed=s,
                                  replace=rep))
            fl.append(max(g[501:].mean(), 0.0))
        print(f"batch k={k}: steps/epoch={m // k} floor with "
              f"{fl[0]:.2e} without {fl[1]:.2e}  eta*s2/(4k) "
              f"{0.1 * s2 / (4 * k):.2e}")
