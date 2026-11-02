"""Chapter 19: RAdam (Liu et al. 2020), Lookahead (Zhang et al. 2019),
AdaBelief (Zhuang et al. 2020)."""
import numpy as np
from students import x1, x2
from net221 import init, flat, loss_flat, grad_flat, Xs, ys, m
from ch10_perweight import batches, first, LIN
from ch14_adam import OM1, OM2, ZETA
from ch15_schedules import constant, warm

np.set_printoptions(precision=4, suppress=True)
P0 = flat(*init())


# ----------------------- RAdam quantities (symbolic) -----------------------
def q_inf(om2=OM2):
    return 2 / (1 - om2) - 1


def q_t(t, om2=OM2):
    """Length of the simple average that matches the EMA (paper's rho_t)."""
    return q_inf(om2) - 2 * t * om2 ** t / (1 - om2 ** t)


def mean_age(t, om2=OM2):
    """Centre of mass of the bias-corrected EMA weights, by direct sum."""
    a = np.arange(t)
    wts = (1 - om2) * om2 ** a / (1 - om2 ** t)
    return wts @ a


def nu_t(q, qi):
    """Rectification term (paper's r_t); defined for q > 4."""
    return np.sqrt((q - 4) * (q - 2) * qi / ((qi - 4) * (qi - 2) * q))


def var_approx(q, s2=1.0):
    """First-order variance of psi = 1/sqrt(rhat), Equation (19.var)."""
    return q / (2 * (q - 2) * (q - 4) * s2)


def mc_var(ts, om2=OM2, n=200000, seed=9):
    """Monte Carlo variance of 1/sqrt(rhat_t) for i.i.d. N(0,1) gradients."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    out = {}
    for t in range(1, max(ts) + 1):
        g = rng.standard_normal(n)
        r = om2 * r + (1 - om2) * g * g
        if t in ts:
            out[t] = np.var(1 / np.sqrt(r / (1 - om2 ** t)))
    return out


# ------------------------------ optimisers ------------------------------
def adaptive(g, eta, steps, p=P0, kind="adam", om1=OM1, om2=OM2,
             zeta=ZETA, thr=4.0, log=None):
    """Adam, RAdam or AdaBelief with eta(t); returns the losses."""
    p = p.copy()
    mt, r = np.zeros(len(p)), np.zeros(len(p))
    qi = q_inf(om2)
    hist = [loss_flat(p)]
    for t in range(1, steps + 1):
        gt = g(p)
        mt = om1 * mt + (1 - om1) * gt
        if kind == "adabelief":
            r = om2 * r + (1 - om2) * (gt - mt) ** 2 + zeta
        else:
            r = om2 * r + (1 - om2) * gt * gt
        mh = mt / (1 - om1 ** t)
        rh = r / (1 - om2 ** t)
        if kind == "radam":
            q = q_t(t, om2)
            if q > thr:
                nu = nu_t(q, qi)
                d = nu * mh / (np.sqrt(rh) + zeta)
            else:
                nu, d = None, mh
        else:
            nu, d = None, mh / (np.sqrt(rh) + zeta)
        if log is not None:
            log.append(dict(t=t, g=gt.copy(), m=mh.copy(), r=rh.copy(),
                            nu=nu, d=d.copy(), eta=eta(t - 1)))
        p = p - eta(t - 1) * d
        hist.append(loss_flat(p))
        if not np.isfinite(hist[-1]) or hist[-1] > 1e6:
            hist += [np.inf] * (steps - t)
            break
    return np.array(hist)


def lookahead_adam(g, eta, steps, p=P0, K=5, alpha=0.5, om1=OM1, om2=OM2,
                   zeta=ZETA):
    """Lookahead around Adam. The Adam moments carry over between cycles.
    The loss of the slow weights is recorded after every gradient step."""
    phi = p.copy()
    w = p.copy()
    mt, r = np.zeros(len(p)), np.zeros(len(p))
    hist = [loss_flat(phi)]
    for t in range(1, steps + 1):
        gt = g(w)
        mt = om1 * mt + (1 - om1) * gt
        r = om2 * r + (1 - om2) * gt * gt
        mh = mt / (1 - om1 ** t)
        rh = r / (1 - om2 ** t)
        w = w - eta(t - 1) * mh / (np.sqrt(rh) + zeta)
        if t % K == 0:
            phi = phi + alpha * (w - phi)
            w = phi.copy()
        hist.append(loss_flat(phi))
        if not np.isfinite(hist[-1]) or hist[-1] > 1e6:
            hist += [np.inf] * (steps - t)
            break
    return np.array(hist)


TW = 100
METHODS = ("Adam", "Adam + warm-up", "RAdam", "Lookahead(Adam)", "AdaBelief")


def run(name, eta, steps, b=None, seed=9):
    if b is None:
        g = grad_flat
    else:
        gen = batches(b, seed)
        g = lambda p: grad_flat(p, next(gen))
    f = constant(eta)
    if name == "Adam":
        return adaptive(g, f, steps)
    if name == "Adam + warm-up":
        return adaptive(g, warm(constant(eta), TW, eta), steps)
    if name == "RAdam":
        return adaptive(g, f, steps, kind="radam")
    if name == "AdaBelief":
        return adaptive(g, f, steps, kind="adabelief")
    return lookahead_adam(g, f, steps)


# ------------------------- Lookahead on a quadratic -------------------------
def la_ratio(a, K, alpha):
    """Steady variance of slow weights over that of SGD, one dimension."""
    return alpha * (1 + a ** K) / (2 - alpha * (1 - a ** K))


def la_sim(h=1.0, eta=0.5, sig=1.0, K=5, alpha=0.5, cycles=200000, seed=9):
    """Simulated variance on L = h (w - c)^2 / 2 with c ~ N(0, sig^2)."""
    rng = np.random.default_rng(seed)
    w, ws = 0.0, []
    phi, ps = 0.0, []
    for _ in range(cycles):
        c = sig * rng.standard_normal(K)
        for i in range(K):                      # plain SGD, same noise
            w = w - eta * h * (w - c[i])
            ws.append(w)
        u = phi
        for i in range(K):
            u = u - eta * h * (u - c[i])
        phi = phi + alpha * (u - phi)
        ps.append(phi)
    return np.var(ws[1000:]), np.var(ps[200:])


# ------------------------- hand examples ---------------------------------
def lin_parts():
    """Standardised linear model: H = Xs'Xs/m, b = Xs'ys/m."""
    return Xs.T @ Xs / m, Xs.T @ ys / m


def lookahead_hand(eta=1.3, K=3, alpha=0.5):
    Hs, bs = lin_parts()
    gl = lambda w: Hs @ w - bs
    Ll = lambda w: 0.5 * w @ Hs @ w - bs @ w + ys @ ys / (2 * m)
    phi = np.zeros(3)
    w = phi.copy()
    rows = [("start", w.copy(), Ll(w))]
    for c in range(2):
        for i in range(K):
            w = w - eta * gl(w)
            rows.append((f"cycle {c + 1}, inner {i + 1}", w.copy(), Ll(w)))
        phi = phi + alpha * (w - phi)
        w = phi.copy()
        rows.append((f"cycle {c + 1}, slow", phi.copy(), Ll(phi)))
    gdw = np.zeros(3)
    gdrows = []
    for t in range(2 * K):
        gdw = gdw - eta * gl(gdw)
        gdrows.append((t + 1, gdw.copy(), Ll(gdw)))
    return Hs, bs, rows, gdrows


if __name__ == "__main__":
    print("== RAdam: q_inf and q_t for omega2 = 0.999")
    qi = q_inf()
    print(f"q_inf = {qi:.0f}")
    for t in range(1, 11):
        q = q_t(t)
        nu = nu_t(q, qi) if q > 4 else float("nan")
        print(f"t={t:2d}  q_t={q:.4f}  1+2*age={1 + 2 * mean_age(t):.4f}"
              f"  nu_t={nu:.4f}")
    for t in (20, 50, 100, 200, 500, 1000, 2000, 5000):
        q = q_t(t)
        print(f"t={t:5d}  q_t={q:8.2f}  nu_t={nu_t(q, qi):.4f}")
    print("first t with q_t > 4:", next(t for t in range(1, 99)
                                        if q_t(t) > 4),
          " q_t > 5:", next(t for t in range(1, 99) if q_t(t) > 5))
    print("nu_t > 0.5 from t =", next(t for t in range(5, 9999)
                                      if nu_t(q_t(t), qi) > 0.5),
          "; > 0.9 from t =", next(t for t in range(5, 99999)
                                   if nu_t(q_t(t), qi) > 0.9))

    print("\n== Variance of psi = 1/sqrt(rhat): Monte Carlo vs (19.var)")
    ts = (5, 6, 8, 10, 20, 50, 100, 500)
    mc = mc_var(ts)
    for t in ts:
        q = q_t(t)
        print(f"t={t:3d}  q_t={q:7.2f}  MC={mc[t]:.4f}"
              f"  approx={var_approx(q):.4f}"
              f"  rectified MC={nu_t(q, qi) ** 2 * mc[t]:.6f}")
    print(f"approx at q_inf: {var_approx(qi):.6f}")

    print("\n== Hand example (a): RAdam vs Adam on the net, full batch, "
          "eta = 0.1")
    lr, la = [], []
    adaptive(grad_flat, constant(0.1), 6, kind="radam", log=lr)
    adaptive(grad_flat, constant(0.1), 6, kind="adam", log=la)
    for j in (6, 8):
        print(f"weight index {j}")
        for a, b in zip(lr, la):
            nu = "-" if a["nu"] is None else f"{a['nu']:.4f}"
            print(f" t={a['t']} g={a['g'][j]:+.4f} mhat={a['m'][j]:+.4f}"
                  f" rhat={a['r'][j]:.5f} nu={nu}"
                  f" RAdam move={-0.1 * a['d'][j]:+.5f}"
                  f" Adam move={-0.1 * b['d'][j]:+.5f}")
    print("losses after 6 steps: RAdam",
          f"{adaptive(grad_flat, constant(0.1), 6, kind='radam')[-1]:.4f}",
          "Adam", f"{adaptive(grad_flat, constant(0.1), 6)[-1]:.4f}")

    print("\n== Hand example (b): Lookahead(GD), K = 3, alpha = 0.5, "
          "eta = 1.3, standardised linear model")
    Hs, bs, rows, gdrows = lookahead_hand()
    print("H =\n", Hs, "\nb =", bs)
    print("eigenvalues of H:", np.linalg.eigvalsh(Hs))
    print("w* =", np.linalg.solve(Hs, bs),
          " L* =", f"{0.5 * ys @ ys / m - 0.5 * bs @ np.linalg.solve(Hs, bs):.6f}")
    for name, w, L in rows:
        print(f"{name:18s} w = {w}  L = {L:.4f}")
    for t, w, L in gdrows:
        print(f"GD step {t}  w = {w}  L = {L:.4f}")
    lam = np.linalg.eigvalsh(Hs)[[0, 2]]
    a = 1 - 1.3 * lam
    print("2/lambda_max =", f"{2 / lam[1]:.4f}", " 1 - eta*lambda =", a,
          " per cycle: GD", a ** 3, " Lookahead", 0.5 + 0.5 * a ** 3)

    print("\n== Hand example (c): AdaBelief vs Adam, full batch, eta = 0.1")
    lb, la = [], []
    adaptive(grad_flat, constant(0.1), 4, kind="adabelief", log=lb)
    adaptive(grad_flat, constant(0.1), 4, kind="adam", log=la)
    j = 8
    for a, b in zip(lb, la):
        print(f" t={a['t']} AdaBelief: g={a['g'][j]:+.4f}"
              f" mhat={a['m'][j]:+.4f} shat={a['r'][j]:.5f}"
              f" move={-0.1 * a['d'][j]:+.4f}")
        print(f"      Adam:      g={b['g'][j]:+.4f}"
              f" mhat={b['m'][j]:+.4f} rhat={b['r'][j]:.5f}"
              f" move={-0.1 * b['d'][j]:+.4f}")

    print("\n== Lookahead on L = h(w-c)^2/2: ratio (19.ratio) vs simulation")
    for eta, K, al in ((0.5, 5, 0.5), (0.5, 5, 0.8), (1.5, 5, 0.5),
                       (0.1, 5, 0.5)):
        a = 1 - eta
        vs, vl = la_sim(eta=eta, K=K, alpha=al)
        print(f"eta h={eta} K={K} alpha={al}: formula {la_ratio(a, K, al):.4f}"
              f"  sim {vl / vs:.4f}  (V_SGD sim {vs:.4f},"
              f" exact {eta ** 2 / (1 - a * a):.4f})")

    print("\n== Training (d), 2000 steps, steps to linear L*, final loss")
    for b in (None, 2):
        for eta in (0.01, 0.03, 0.1, 0.3, 1.0):
            for name in METHODS:
                h = run(name, eta, 2000, b)
                tail = h[1001:]
                print(f"b={b} eta={eta:<5} {name:16s} hit={first(h, LIN)}"
                      f"  L(2000)={h[-1]:.2e}  max L={np.max(h[1:]):.3f}"
                      f"  mean L(1001-2000)={np.mean(tail):.2e}")
