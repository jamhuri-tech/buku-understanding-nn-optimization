"""Chapter 8: Nesterov's accelerated gradient on the students."""
import numpy as np
from students import X, y, m, x1, x2, w_star, loss
from ch06_momentum import make, tune, Xc, lessard_grad

Lstar = loss(w_star)


def sc_gamma(kappa):
    """Constant momentum for strongly convex losses."""
    s = np.sqrt(kappa)
    return (s - 1) / (s + 1)


def nag(g, eta, gam, steps, w=None, L=None, restart=False, log=None):
    """Nesterov from w (default 0), w_{-1} = w_0.

    gam is a number (constant momentum) or None (schedule (t-1)/(t+2)).
    With restart=True the schedule counter returns to 1 whenever the
    loss rises (function restart).  Returns the iterates.
    """
    w = np.zeros(3) if w is None else np.array(w, float)
    prev = w.copy()
    path = [w.copy()]
    k = 0                      # steps since the last restart
    for _ in range(steps):
        gt = gam if gam is not None else (k - 1) / (k + 2) if k > 0 else 0.0
        u = w + gt * (w - prev)
        new = u - eta * g(u)
        if restart and L(new) > L(w):
            if log is not None:
                log.append(len(path))
            k = 0
            u = w
            new = w - eta * g(w)
        w, prev = new, w
        k += 1
        path.append(w.copy())
    return path


def hb(g, eta, gam, steps, w=None):
    w = np.zeros(3) if w is None else np.array(w, float)
    prev = w.copy()
    path = [w.copy()]
    for _ in range(steps):
        w, prev = w - eta * g(w) + gam * (w - prev), w
        path.append(w.copy())
    return path


def first_hit(path, L, target):
    for t, p in enumerate(path):
        if L(p) <= target:
            return t
    return None


def nag_roots(eta, gam, lam):
    q = 1 - eta * lam
    return np.roots([1.0, -(1 + gam) * q, gam * q])


def lessard_nag(x0, steps):
    eta, gam = 1 / 25, sc_gamma(25.0)
    x, prev, out = x0, x0, [x0]
    for _ in range(steps):
        u = x + gam * (x - prev)
        x, prev = u - eta * lessard_grad(u), x
        out.append(x)
    return np.array(out)


def ce_problem():
    t_lab = (y >= 65).astype(float)
    rho = 0.01
    sig = lambda z: 1 / (1 + np.exp(-z))
    L = lambda w: (np.mean(np.logaddexp(0, X @ w) - t_lab * (X @ w))
                   + 0.5 * rho * w[1:] @ w[1:])
    G = lambda w: X.T @ (sig(X @ w) - t_lab) / m + rho * np.r_[0, w[1:]]
    H = X.T @ X / m
    beta = np.linalg.eigvalsh(H)[-1] / 4 + rho
    w = np.zeros(3)
    for _ in range(50):                       # Newton for the minimiser
        p = sig(X @ w)
        Hc = X.T @ (X * (p * (1 - p))[:, None]) / m + rho * np.diag([0, 1, 1])
        w = w - np.linalg.solve(Hc, G(w))
    p = sig(X @ w)
    Hc = X.T @ (X * (p * (1 - p))[:, None]) / m + rho * np.diag([0, 1, 1])
    return L, G, beta, w, L(w), np.linalg.eigvalsh(Hc)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    np.seterr(all="ignore")

    # toy quadratic mu = 1, beta = 25
    k = 25.0
    gm = sc_gamma(k)
    print(f"toy: gamma = {gm:.4f}, rate = {1-1/np.sqrt(k):.4f}")
    for lam in (1.0, 9.0, 25.0):
        r = nag_roots(1 / 25, gm, lam)
        print(f"  lambda={lam:4.0f}: q = {1-lam/25:.4f}, roots {np.round(r, 4)},"
              f" |r| = {abs(r).max():.4f}")
    for name, r in (("NAG", 0.8), ("HB", 2 / 3), ("GD", 24 / 26)):
        print(f"  steps for 1e-6 {name}: {np.log(1e6)/-np.log(r):.1f}")

    # schedule values
    print("gamma_t = (t-1)/(t+2):", [f"{(t-1)/(t+2):.4f}" for t in range(1, 7)])
    th = [1.0]
    for _ in range(6):
        th.append((1 + np.sqrt(1 + 4 * th[-1] ** 2)) / 2)
    print("theta_t (1983):", np.round(th, 4))
    print("gamma_t (1983):", np.round([(th[i] - 1) / th[i + 1]
                                       for i in range(6)], 4))

    # Lessard function under Nesterov
    xs = lessard_nag(3.3, 200)
    print("Lessard NAG x0=3.3:", np.round(xs[:5], 4))
    print(f"  |x_t| at t=50,100,200: {abs(xs[50]):.2e} {abs(xs[100]):.2e}"
          f" {abs(xs[200]):.2e}")
    worst = max(abs(lessard_nag(x0, 300)[-1]) for x0 in np.linspace(-10, 10, 201))
    print(f"  max |x_300| over starts in [-10,10]: {worst:.2e}")

    # raw students
    H, ws, L, g = make(X)
    lam = np.linalg.eigvalsh(H)
    mu, be = lam[0], lam[-1]
    kap = be / mu
    gm = sc_gamma(kap)
    eta = 1 / be
    print(f"\nraw: mu = {mu:.4f}, beta = {be:.4f}, kappa = {kap:.2f},"
          f" sqrt kappa = {np.sqrt(kap):.4f}")
    print(f"  eta = 1/beta = {eta:.6f}, gamma = {gm:.5f},"
          f" rate = {1-1/np.sqrt(kap):.5f}")

    # two steps by hand
    w0 = np.zeros(3)
    g0 = g(w0)
    w1 = w0 - eta * g0
    u1 = w1 + gm * (w1 - w0)
    g1 = g(u1)
    w2 = u1 - eta * g1
    print("  g0 =", g0, " w1 =", w1, f" L(w1) = {L(w1):.4f}")
    print("  H =\n", H, "\n  Xty/m =", X.T @ y / m)
    print("  u1 =", u1, " H u1 =", H @ u1, " g(u1) =", g1)
    print("  eta g(u1) =", eta * g1)
    print("  w2 =", w2, f" L(w2) = {L(w2):.4f}, L(u1) = {L(u1):.4f}")
    # heavy ball second step for comparison, same eta and gamma
    w2hb = w1 - eta * g(w1) + gm * (w1 - w0)
    print("  HB same eta,gamma: g(w1) =", g(w1), " w2 =", w2hb,
          f" L = {L(w2hb):.4f}")

    # per-mode factors
    eHB, gHB = tune(mu, be)
    eGD = 2 / (mu + be)
    print("\nlambda    GD       HB       NAG q     NAG |r|")
    for l in lam:
        r = nag_roots(eta, gm, l)
        rh = np.roots([1.0, -(1 + gHB - eHB * l), gHB])
        q = 1 - eta * l
        print(f"{l:8.4f} {1-eGD*l:8.5f} {abs(rh).max():8.5f}"
              f" {q:9.5f} {abs(r).max():8.5f}"
              f"   (1+g)^2q^2 = {(1+gm)**2*q*q:.4f}, 4gq = {4*gm*q:.4f}")

    # steps to target
    target = Lstar + 0.01
    print(f"\ntarget = {target:.4f}")
    N = 20000
    for name, A in (("raw", X), ("centred", Xc)):
        H, ws, L, g = make(A)
        lam = np.linalg.eigvalsh(H)
        mu, be = lam[0], lam[-1]
        kap = be / mu
        e1, g1_ = tune(mu, be)
        n = min(N, 200)
        runs = {
            "GD": hb(g, 2 / (mu + be), 0.0, N),
            "HB": hb(g, e1, g1_, N),
            "NAG-sc": nag(g, 1 / be, sc_gamma(kap), N),
            "NAG-cvx": nag(g, 1 / be, None, N),
            "NAG-restart": nag(g, 1 / be, None, N, L=L, restart=True),
        }
        print(name, f"kappa = {kap:.2f}, gamma_sc = {sc_gamma(kap):.5f}",
              {k_: first_hit(p, L, target) for k_, p in runs.items()})
        tg6 = Lstar + 1e-6
        print("  to L*+1e-6:", {k_: first_hit(p, L, tg6) for k_, p in runs.items()})
        print(f"  predicted NAG-sc: "
              f"{np.log((L(np.zeros(3))-Lstar)/0.01)/(-2*np.log(1-1/np.sqrt(kap))):.1f}")

    # gap at t = 100, 1000 for the raw runs; bound for convex schedule
    H, ws, L, g = make(X)
    be = np.linalg.eigvalsh(H)[-1]
    R2 = ws @ ws
    P = nag(g, 1 / be, None, 3000)
    Q = hb(g, 1 / be, 0.0, 3000)
    for t in (10, 100, 1000, 3000):
        print(f"  t={t}: NAG-cvx gap {L(P[t])-Lstar:.4e},"
              f" bound {2*be*R2/(t+1)**2:.4e};"
              f" GD(1/beta) gap {L(Q[t])-Lstar:.4e},"
              f" bound {be*R2/(2*t):.4e}")
    print(f"  ||w*||^2 = {R2:.4f}")
    # restarts in the raw run
    gs = sc_gamma(be / np.linalg.eigvalsh(H)[0])
    print(f"  (t-1)/(t+2) passes {gs:.5f} at t = {(1+2*gs)/(1-gs):.2f}")
    rst = []
    nag(g, 1 / be, None, 400, L=L, restart=True, log=rst)
    print("  restarts at t (w_t is a plain gradient step):", rst)
    print("  first loss rises of NAG-cvx (raw):",
          [t for t in range(1, 400) if L(P[t]) > L(P[t - 1])][:6])
    # misestimated mu
    lam = np.linalg.eigvalsh(H)
    for f in (0.1, 10.0):
        kk = be / (f * lam[0])
        p = nag(g, 1 / be, sc_gamma(kk), N)
        print(f"  mu x {f}: gamma = {sc_gamma(kk):.5f}, steps "
              f"{first_hit(p, L, target)}")

    # cross-entropy
    Lc, Gc, bc, wc, Lcs, eigc = ce_problem()
    print(f"\nCE: beta = {bc:.4f}, w* =", wc, f" L* = {Lcs:.6f}")
    print(f"  Hessian eig at w*: {eigc}, kappa = {eigc[-1]/eigc[0]:.1f}")
    T = 200000
    tg = Lcs + 1e-6
    for name, gam, rs in (("GD", 0.0, False), ("NAG-cvx", None, False),
                          ("NAG-restart", None, True)):
        w, prev, k_ = np.zeros(3), np.zeros(3), 0
        hit = None
        Lw = Lc(w)
        for t in range(T):
            if Lw <= tg:
                hit = t
                break
            gt = gam if gam is not None else ((k_ - 1) / (k_ + 2) if k_ > 0 else 0.0)
            u = w + gt * (w - prev)
            new = u - Gc(u) / bc
            Ln = Lc(new)
            if rs and Ln > Lw:
                k_, new = 0, w - Gc(w) / bc
                Ln = Lc(new)
            w, prev, Lw = new, w, Ln
            k_ += 1
        print(f"  {name}: steps to L*+1e-6 = {hit}")
