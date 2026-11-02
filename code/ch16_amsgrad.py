"""Chapter 16: AMSGrad (Reddi, Kale and Kumar 2018) and the gap in Adam."""
import numpy as np
from students import loss, grad
from net221 import init, flat, loss_flat, grad_flat
from ch10_perweight import batches, first, LIN
from ch12_adagrad import fmt
from ch14_adam import adam, LS, OM1, OM2, ZETA

np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------
# The online counterexample on [-1, 1]
# ---------------------------------------------------------------------
def online(big, small, P, T, eta, om1=0.0, om2=0.1, kind="rmsprop",
           x0=1.0, decay=True):
    """f_t(x) = big*x when t mod P = 0, small*x otherwise; t starts at 0.

    kind: 'sgd', 'rmsprop' (= Adam without correction) or 'amsgrad'.
    Rates eta_t = eta/sqrt(t+1) (decay) or eta.  No bias correction and
    zeta = 0, as in Reddi et al.  The iterate is clipped to [-1, 1].
    Returns arrays x_0..x_T, g, r, rmax, eta_t, move (before clipping).
    """
    x, m, r, rm = x0, 0.0, 0.0, 0.0
    xs, gs, rs, rms, es, mv = [x], [], [], [], [], []
    for t in range(T):
        g = big if t % P == 0 else small
        et = eta / np.sqrt(t + 1) if decay else eta
        if kind == "sgd":
            s = -et * g
        else:
            m = om1 * m + (1 - om1) * g
            r = om2 * r + (1 - om2) * g * g
            rm = max(rm, r)
            s = -et * m / np.sqrt(rm if kind == "amsgrad" else r)
        x = min(1.0, max(-1.0, x + s))
        xs.append(x); gs.append(g); rs.append(r); rms.append(rm)
        es.append(et); mv.append(s)
    return tuple(np.array(a) for a in (xs, gs, rs, rms, es, mv))


def regret(xs, gs):
    """Average regret R_T/T against the best fixed point of [-1, 1]."""
    paid = np.sum(gs * xs[:-1])
    best = min(np.sum(gs) * z for z in (-1.0, 1.0))
    return (paid - best) / len(gs)


def gamma(rd, es):
    """Gamma_t = sqrt(r_t)/eta_t - sqrt(r_{t-1})/eta_{t-1}, t >= 1."""
    q = np.sqrt(rd) / es
    return q[1:] - q[:-1]


THM = dict(big=3.0, small=-1.0, P=3, eta=0.5, om2=0.1)       # C = 3
SYN = dict(big=1010.0, small=-10.0, P=101)                  # Reddi Sec. 5
SYN_RUNS = {"SGD": dict(kind="sgd", eta=0.05),
            "RMSProp": dict(kind="rmsprop", eta=0.5, om1=0.0, om2=0.99),
            "Adam": dict(kind="rmsprop", eta=0.5, om1=0.9, om2=0.99),
            "AMSGrad": dict(kind="amsgrad", eta=0.5, om1=0.9, om2=0.99)}


def thm_runs(T=300):
    return {k: online(T=T, kind=k, **THM)
            for k in ("sgd", "rmsprop", "amsgrad")}


def syn_runs(cycles=2000):
    return {k: online(T=101 * cycles, **SYN, **a) for k, a in SYN_RUNS.items()}


# ---------------------------------------------------------------------
# AMSGrad for the students, in the library form (max of raw r_t, then
# Adam's bias corrections), as in torch.optim.Adam(amsgrad=True)
# ---------------------------------------------------------------------
def amsgrad(g, L, w, eta, om1=OM1, om2=OM2, steps=1, zeta=ZETA,
            target=None, log=None):
    w = w.copy()
    mt, r, rm = np.zeros(len(w)), np.zeros(len(w)), np.zeros(len(w))
    hist, hit = [L(w)], None
    for t in range(steps):
        gt = g(w)
        mt = om1 * mt + (1 - om1) * gt
        r = om2 * r + (1 - om2) * gt * gt
        rm = np.maximum(rm, r)
        mh = mt / (1 - om1 ** (t + 1))
        den = np.sqrt(rm / (1 - om2 ** (t + 1))) + zeta
        s = -eta * mh / den
        if log is not None:
            log.append((r.copy(), rm.copy()))
        w = w + s
        hist.append(L(w))
        if hit is None and target is not None and hist[-1] <= target:
            hit = t + 1
    return w, hit, hist


def linear_pair(steps=20000):
    w0, T = np.zeros(3), LS + 0.01
    out = {}
    for eta in (0.1, 1.0):
        out[f"Adam {eta}"] = adam(grad, loss, w0, eta, steps=steps, target=T)
        out[f"AMSGrad {eta}"] = amsgrad(grad, loss, w0, eta, steps=steps,
                                        target=T)
    return out


def net_pair(eta, steps=2000, b=None):
    """Adam and AMSGrad on the 2-2-1 net; b = mini-batch size or None."""
    p0 = flat(*init())
    out = {}
    for k, f in (("Adam", adam), ("AMSGrad", amsgrad)):
        if b is None:
            gfun = grad_flat
        else:
            gen = batches(b)
            gfun = lambda p, gen=gen: grad_flat(p, next(gen))
        log = []
        if f is amsgrad:
            res = f(gfun, loss_flat, p0, eta, steps=steps, log=log)
        else:
            res = f(gfun, loss_flat, p0, eta, steps=steps)
        out[k] = (res[0], res[2], log)
    return out


if __name__ == "__main__":
    C, w2 = THM["big"], THM["om2"]
    print(f"== Counterexample, C = {C:g}, omega2 = 1/(1+C^2) = {w2:g},"
          f" omega1 = 0, eta = {THM['eta']}, eta_t = eta/sqrt(t+1), x0 = 1")
    print(f"sqrt(1-omega2) = {np.sqrt(1 - w2):.4f}"
          f"   omega2*C^2 + 1 - omega2 = {w2 * C * C + 1 - w2:.4f}"
          f" = 2(1-omega2)")
    print(f"time average of g over a cycle = {(C - 2) / 3:.4f}")
    R = thm_runs()
    for k in ("rmsprop", "amsgrad"):
        xs, gs, rs, rms, es, mv = R[k]
        print(f"-- {k}")
        print("  t   g     eta_t     r_t     rmax_t   move     x_{t+1}")
        for t in range(6):
            print(f"  {t}  {gs[t]:+.0f}  {es[t]:.4f}  {rs[t]:.4f}"
                  f"  {rms[t]:.4f}  {mv[t]:+.4f}  {xs[t + 1]:+.4f}")
        rd = rms if k == "amsgrad" else rs
        print("  Gamma_1..5 =", np.round(gamma(rd, es)[:5], 4))
    for k, (xs, gs, *_rest) in R.items():
        print(f"{k:8s} x_300 = {xs[-1]:+.4f}  R_T/T = {regret(xs, gs):.4f}")
    xs = R["rmsprop"][0]
    print("RMSProp x at t = 3k, k = 1..100 all equal 1:",
          bool(np.all(xs[3:301:3] == 1.0)))
    print(f"lower bound (2C-4)/3 = {(2 * C - 4) / 3:.4f}")
    xs, gs, rs, rms, es, mv = online(T=30000, kind="rmsprop", **THM)
    print(f"RMSProp T = 30000: R_T/T = {regret(xs, gs):.4f}")
    xa, ga, *_r = online(T=30000, kind="amsgrad", **THM)
    print(f"AMSGrad T = 30000: R_T/T = {regret(xa, ga):.4f}"
          f"  x_T = {xa[-1]:+.4f}  rmax = {_r[1][-1]:.4f}")
    print("-- per-cycle bounds (k >= 1)")
    dn = np.sqrt(1 / (1 - w2)); up = np.sqrt(1 / (2 * (1 - w2)))
    print(f"  down move <= {dn:.4f} eta_t,  each up move >= {up:.4f} eta_t,"
          f"  two up >= {2 * up:.4f}")

    print("\n== Reddi et al. synthetic: g = 1010 when t mod 101 = 0,"
          " else -10; eta_t = eta/sqrt(t+1)")
    S = syn_runs()
    for k, (xs, gs, *_r) in S.items():
        cm = xs[:-1].reshape(-1, 101).mean(1)
        print(f"{k:8s} cycle means at 1,10,100,1000,2000:"
              f" {cm[0]:+.3f} {cm[9]:+.3f} {cm[99]:+.3f} {cm[999]:+.3f}"
              f" {cm[-1]:+.3f}   R_T/T = {regret(xs, gs):.4f}")

    print("\n== Linear model, raw features, steps to L* + 0.01")
    for k, r in linear_pair().items():
        h = r[2]
        print(f"{k:12s} {fmt(r[1])}   gap(20000) = {h[-1] - LS:.3e}"
              f"   max gap 10001-20000 = {max(h[10001:]) - LS:.3e}")
    log = []
    amsgrad(grad, loss, np.zeros(3), 0.1, steps=5001, log=log)
    act = next(t for t, (r, rm) in enumerate(log) if np.any(rm > r))
    print(f"AMSGrad 0.1: the max first acts at t = {act}")
    for t in (act - 1, act):
        print(f"  t={t}  r = {np.array2string(log[t][0], precision=4)}"
              f"  rmax = {np.array2string(log[t][1], precision=4)}")
    for t in (100, 1000, 5000):
        r, rm = log[t]
        print(f"  t={t:4d}  sqrt(rmax/r) = {np.sqrt(rm / r)}")

    print("\n== 2-2-1 network, full batch")
    for eta in (0.01, 0.03):
        out = net_pair(eta)
        for k, (p, h, log) in out.items():
            print(f"eta={eta} {k:8s} to L* {fmt(first(h, LIN))}"
                  f"  to 1e-4 {fmt(first(h, 1e-4))}  L(2000) = {h[-1]:.2e}")
        r, rm = out["AMSGrad"][2][-1]
        print(f"   AMSGrad at t=1999: max sqrt(rmax/r) = "
              f"{np.sqrt(rm / r).max():.3f}")
        d = np.abs(out["Adam"][0] - out["AMSGrad"][0]).max()
        print(f"   largest weight difference at t=2000 = {d:.2e}")

    print("\n== 2-2-1 network, mini-batches of two, eta = 0.03,"
          " full-batch loss")
    out = net_pair(0.03, b=2)
    for k, (p, h, log) in out.items():
        print(f"{k:8s} L(100) = {h[100]:.2e}  L(500) = {h[500]:.2e}"
              f"  L(2000) = {h[-1]:.2e}")
        print(f"{k:8s} to L* {fmt(first(h, LIN))}"
              f"  max L(1000..2000) = {max(h[1000:]):.2e}")
    r, rm = out["AMSGrad"][2][-1]
    ratio = np.sqrt(rm / r)
    print(f"AMSGrad at t=1999: sqrt(rmax/r) min {ratio.min():.2f}"
          f" max {ratio.max():.2f}")
