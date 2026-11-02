"""Chapter 6: heavy-ball momentum on the students."""
import numpy as np
from students import X, y, m, x1, x2, w_star, loss

Xc = np.column_stack([np.ones(m), x1 - x1.mean(), x2 - x2.mean()])
Lstar = loss(w_star)


def make(A):
    H = A.T @ A / m
    ws = np.linalg.solve(H, A.T @ y / m)
    L = lambda w: (A @ w - y) @ (A @ w - y) / (2 * m)
    g = lambda w: A.T @ (A @ w - y) / m
    return H, ws, L, g


def tune(mu, be):
    """Polyak's tuning for eigenvalues in [mu, be]."""
    eta = 4 / (np.sqrt(be) + np.sqrt(mu)) ** 2
    k = be / mu
    gam = ((np.sqrt(k) - 1) / (np.sqrt(k) + 1)) ** 2
    return eta, gam


def hb(g, eta, gam, steps, w=None):
    """Heavy ball from w (default 0) with w_{-1} = w_0; returns iterates."""
    w = np.zeros(3) if w is None else np.array(w, float)
    prev = w.copy()
    path = [w.copy()]
    for _ in range(steps):
        w, prev = w - eta * g(w) + gam * (w - prev), w
        path.append(w.copy())
    return path


def count(A, eta, gam, target, maxit=10**6):
    """Steps of heavy ball (gam = 0 gives GD) until L <= target."""
    H, ws, L, g = make(A)
    w = np.zeros(3)
    prev = w.copy()
    for t in range(maxit):
        if L(w) <= target:
            return t
        w, prev = w - eta * g(w) + gam * (w - prev), w
    return None


def roots(eta, gam, lam):
    return np.roots([1.0, -(1 + gam - eta * lam), gam])


def lessard_grad(x):
    if x < 1:
        return 25 * x
    if x < 2:
        return x + 24
    return 25 * x - 24


def lessard(x0, steps):
    eta, gam = tune(1.0, 25.0)
    x, prev, out = x0, x0, [x0]
    for _ in range(steps):
        x, prev = x - eta * lessard_grad(x) + gam * (x - prev), x
        out.append(x)
    return np.array(out)


def msgd(eta, gam, epochs, k=1, seed=0):
    """Mini-batch heavy ball on centred features; gaps after each epoch."""
    H, ws, L, g = make(Xc)
    rng = np.random.default_rng(seed)
    w, v = np.zeros(3), np.zeros(3)
    gaps = [L(w) - Lstar]
    for _ in range(epochs):
        order = rng.integers(0, m, m)
        for j in range(0, m, k):
            B = order[j:j + k]
            gb = Xc[B].T @ (Xc[B] @ w - y[B]) / len(B)
            v = gam * v - eta * gb
            w = w + v
        gaps.append(L(w) - Lstar)
    return np.array(gaps)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    np.seterr(all="ignore")
    target = Lstar + 0.01
    print(f"L* = {Lstar:.4f}, target = {target:.4f}")

    # tuning for the toy range [1, 25]
    eta, gam = tune(1.0, 25.0)
    print(f"toy mu=1 beta=25: eta = {eta:.4f}, gamma = {gam:.4f}, "
          f"rate HB = {np.sqrt(gam):.4f}, rate GD = {24/26:.4f}")
    print(f"  steps for 1e-6: HB {np.log(1e6)/-np.log(np.sqrt(gam)):.1f}"
          f", GD {np.log(1e6)/-np.log(24/26):.1f}")

    # Lessard counterexample
    xs = lessard(3.3, 400)
    print("Lessard x0=3.3 first iterates:", np.round(xs[:5], 4))
    print("  last 6 iterates:", np.round(xs[-6:], 4))
    for x0 in (3.0, 3.07, 3.3, 3.46, 3.5):
        xs = lessard(x0, 2000)
        print(f"  x0={x0}: max |x| over last 100 = {np.abs(xs[-100:]).max():.4f}")

    print("\nfeatures  mu       beta     kappa    eta_HB   gamma   "
          "rate_HB  rate_GD")
    for name, A in (("raw", X), ("centred", Xc)):
        H, *_ = make(A)
        lam = np.linalg.eigvalsh(H)
        mu, be = lam[0], lam[-1]
        k = be / mu
        e, gm = tune(mu, be)
        print(f"{name:8s} {mu:8.4f} {be:8.4f} {k:8.2f} {e:8.5f} {gm:8.5f}"
              f" {np.sqrt(gm):8.5f} {(k-1)/(k+1):8.5f}")

    # per-mode factors, raw features
    H, ws, L, g = make(X)
    lam = np.linalg.eigvalsh(H)
    mu, be = lam[0], lam[-1]
    e, gm = tune(mu, be)
    eg = 2 / (mu + be)
    print(f"\nraw: eta = {e:.5f}, gamma = {gm:.5f}, sqrt(gamma) = {np.sqrt(gm):.5f}")
    print("lambda    eta*lam   1+g-eta*lam   disc        |r1|     |r2|"
          "   GD factor")
    for l in lam:
        a = 1 + gm - e * l
        r = roots(e, gm, l)
        print(f"{l:8.4f} {e*l:9.5f} {a:12.5f} {a*a-4*gm:11.2e}"
              f" {abs(r[0]):8.5f} {abs(r[1]):8.5f} {1-eg*l:9.5f}")
    print(f"(1-sqrt g)^2 = {(1-np.sqrt(gm))**2:.5f}, "
          f"(1+sqrt g)^2 = {(1+np.sqrt(gm))**2:.5f}, "
          f"eta_GD = {eg:.5f}, 2(1+g)/beta = {2*(1+gm)/be:.5f}")

    # first two steps by hand, raw features
    w0 = np.zeros(3)
    g0 = g(w0)
    v1 = -e * g0
    w1 = w0 + v1
    g1 = g(w1)
    v2 = gm * v1 - e * g1
    w2 = w1 + v2
    print("\nhand steps (raw, eta, gamma as above)")
    print("g0 =", g0, " v1 = w1 =", v1, f" L(w1) = {L(w1):.4f}")
    print("g1 =", g1, " gamma*v1 =", gm * v1, " -eta*g1 =", -e * g1)
    print("v2 =", v2, " w2 =", w2, f" L(w2) = {L(w2):.4f}")
    print(f"L(0) = {L(w0):.4f}")
    wg1 = w0 - eg * g0
    wg2 = wg1 - eg * g(wg1)
    print(f"GD eta={eg:.5f}: L(w1) = {L(wg1):.4f}, L(w2) = {L(wg2):.4f}")

    print("\nsteps to reach the target loss")
    print("features   GD 2/(mu+beta)   HB tuned   kappa-ratio   pred GD  pred HB")
    for name, A in (("raw", X), ("centred", Xc)):
        H, *_ = make(A)
        lam = np.linalg.eigvalsh(H)
        mu, be = lam[0], lam[-1]
        k = be / mu
        e, gm = tune(mu, be)
        n1 = count(A, 2 / (mu + be), 0.0, target)
        n2 = count(A, e, gm, target)
        gap0 = loss(np.zeros(3)) - Lstar
        p1 = np.log(gap0 / 0.01) / (-2 * np.log((k - 1) / (k + 1)))
        p2 = np.log(gap0 / 0.01) / (-2 * np.log(np.sqrt(gm)))
        print(f"{name:9s} {n1:10d} {n2:12d} {n1/n2:12.1f} {p1:10.1f} {p2:8.1f}")

    # sensitivity to the estimate of mu (raw features)
    H, ws, L, g = make(X)
    lam = np.linalg.eigvalsh(H)
    mu, be = lam[0], lam[-1]
    print("\nraw, tuned for a wrong mu (beta exact)")
    print("mu_est     eta       gamma     max|r|   steps")
    for f in (0.25, 1.0, 4.0, 16.0):
        e, gm = tune(f * mu, be)
        rr = max(max(abs(roots(e, gm, l))) for l in lam)
        print(f"{f*mu:8.5f} {e:8.5f} {gm:8.5f} {rr:9.5f} {count(X, e, gm, target)}")
    print("raw, tuned for a wrong beta (mu exact)")
    print("beta_est   eta       gamma     max|r|   peak gap   steps")
    for f in (0.9, 1.0, 1.1, 1.5, 2.0):
        e, gm = tune(mu, f * be)
        rr = max(max(abs(roots(e, gm, l))) for l in lam)
        P = hb(g, e, gm, 3000)
        peak = max(L(p) - Lstar for p in P)
        n = count(X, e, gm, target, maxit=3000)
        print(f"{f*be:8.3f} {e:8.5f} {gm:8.5f} {rr:9.5f} {peak:10.3g}"
              f"   {n if n else 'diverges'}")
    lamq, Q = np.linalg.eigh(H)
    e0 = Q.T @ (np.zeros(3) - ws)
    print("initial gap per mode:", lamq * e0**2 / 2)

    # noisy gradients: heavy ball with mini-batches, centred features
    print("\nsingle-sample heavy ball, centred, floor = mean gap epochs 501-1000")
    print("eta     gamma  eta/(1-gamma)  floor")
    for eta, gam in ((0.02, 0.0), (0.002, 0.9), (0.02, 0.9)):
        G = np.mean([msgd(eta, gam, 1000, seed=s) for s in range(20)], axis=0)
        print(f"{eta:6.3f} {gam:6.2f} {eta/(1-gam):10.3f}   {G[501:].mean():.2e}")

    # the slice w0 = w0* used for the path figure
    H, ws, L, g = make(X)
    S, c = H[1:, 1:], ws[1:]
    lam = np.linalg.eigvalsh(S)
    e, gm = tune(lam[0], lam[-1])
    eg = 2 / (lam[0] + lam[-1])
    k = lam[-1] / lam[0]
    print(f"\nslice: mu = {lam[0]:.4f}, beta = {lam[-1]:.4f}, kappa = {k:.2f}")
    print(f"  GD eta = {eg:.5f}, rate {(k-1)/(k+1):.4f}; HB eta = {e:.5f},"
          f" gamma = {gm:.4f}, rate {np.sqrt(gm):.4f}")
    for eta, gam, name in ((eg, 0.0, "GD"), (e, gm, "HB")):
        v = prev = np.array([0.0, 8.0])
        for _ in range(30):
            v, prev = v - eta * S @ (v - c) + gam * (v - prev), v
        print(f"  {name} after 30 steps: distance to minimiser "
              f"{np.linalg.norm(v - c):.4f}")
    print(f"  start distance {np.linalg.norm(np.array([0.0, 8.0]) - c):.4f}")

    # double-root transient of the steep mode, raw features
    H, ws, L, g = make(X)
    lam = np.linalg.eigvalsh(H)
    e, gm = tune(lam[0], lam[-1])
    r = -np.sqrt(gm)
    t = np.arange(200)
    amp = (1 + (1 - r) * t) * abs(r) ** t
    G = [L(p) - Lstar for p in hb(g, e, gm, 200)]
    print(f"\ntransient: 1 - r = {1 - r:.4f}, amplitude peak {amp.max():.3f}"
          f" at t = {t[amp.argmax()]}, predicted gap {amp.max()**2 * 2384.431:.4g}")
    print(f"  measured peak gap {max(G):.4g} at t = {int(np.argmax(G))}")
