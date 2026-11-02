"""Chapter 15: learning-rate schedules, SGDR and cosine (2016), warm-up
(2017), and the learning-rate range test (Smith 2017)."""
import numpy as np
from students import m
from net221 import init, flat, loss_flat, grad_flat, hess_flat
from ch05_stochastic import Xc, Hc, wc, L as Llin, Lstar, y, spread
from ch10_perweight import batches, first, LIN
from ch12_adagrad import fmt
from ch14_adam import OM1, OM2, ZETA

np.set_printoptions(precision=4, suppress=True)
P0 = flat(*init())


# ---------------------------- the schedules ----------------------------
def constant(e0):
    return lambda t: e0


def step_decay(e0, q, tau):
    return lambda t: e0 * q ** (t // tau)


def exponential(e0, q, tau):
    return lambda t: e0 * q ** (t / tau)


def inverse(e0, tau, p=1.0):
    return lambda t: e0 / (1 + t / tau) ** p


def cosine(emax, T, emin=0.0, tw=0):
    """Linear warm-up for tw steps, then one cosine down to emin at T."""
    def f(t):
        if t < tw:
            return emax * (t + 1) / tw
        return emin + 0.5 * (emax - emin) * (1 + np.cos(np.pi * (t - tw)
                                                         / (T - tw)))
    return f


def sgdr(emax, T0, mult=2, emin=0.0):
    """Cosine with warm restarts, periods T0, mult*T0, ..."""
    def f(t):
        Ti = T0
        while t >= Ti:
            t, Ti = t - Ti, Ti * mult
        return emin + 0.5 * (emax - emin) * (1 + np.cos(np.pi * t / Ti))
    return f


def warm(sched, tw, emax):
    """Linear warm-up to emax over tw steps, then sched(t - tw)."""
    return lambda t: emax * (t + 1) / tw if t < tw else sched(t - tw)


# ------------------------- the linear model (a) -------------------------
T_LIN, E0, K = 400, 0.4, 2
LIN_SCHED = {
    "constant 0.4": constant(0.4),
    "constant 0.04": constant(0.04),
    "step": step_decay(E0, 0.1, 100),
    "exponential": exponential(E0, 0.1, 100),
    "1/t": inverse(E0, 10),
    "cosine": cosine(E0, T_LIN),
}
SB2 = spread(wc) / K * (m - K) / (m - 1)   # Var of a batch of two


def lin_run(eta, T=T_LIN, seed=9):
    """Mini-batch SGD from 0 on centred features; each step draws K
    distinct students, independently of the earlier steps.
    Returns the gaps L(w_t) - L* for t = 0..T."""
    rng = np.random.default_rng(seed)
    w = np.zeros(3)
    gaps = [Llin(w) - Lstar]
    for t in range(T):
        B = rng.choice(m, K, replace=False)
        w = w - eta(t) * Xc[B].T @ (Xc[B] @ w - y[B]) / K
        gaps.append(Llin(w) - Lstar)
    return np.array(gaps)


def lin_table(seeds=200):
    """Mean gap over seeds at t = 100, 200, 400; sum of steps; last eta."""
    out = {}
    for k, f in LIN_SCHED.items():
        G = np.mean([lin_run(f, seed=s) for s in range(seeds)], axis=0)
        S = sum(f(t) for t in range(T_LIN))
        out[k] = (G, S, f(T_LIN - 1))
    return out


def mode_split(eta, T=T_LIN):
    """Bias and variance of each eigen-mode, Equations (15.bias), (15.var),
    for full-batch noise with the batch covariance of Var[g_B]."""
    lam, Q = np.linalg.eigh(Hc)
    G = (Xc @ wc - y)[:, None] * Xc
    S = np.cov((G @ Q).T, bias=True).diagonal() / K * (m - K) / (m - 1)
    c = Q.T @ (np.zeros(3) - wc)
    v = np.zeros(3)
    for t in range(T):
        e = eta(t)
        c = (1 - e * lam) * c
        v = (1 - e * lam) ** 2 * v + e ** 2 * S
    return lam, S, 0.5 * lam * c ** 2, 0.5 * lam * v


# ------------------------- the hand example (b) -------------------------
def hand(emax=0.4, tw=4, T=20, steps=8):
    """Cosine with warm-up on the intercept mode of the centred model,
    e_{t+1} = (1 - eta_t) e_t with e_0 = -68.25."""
    f = cosine(emax, T, tw=tw)
    e, rows = -wc[0], []
    for t in range(steps):
        rows.append((t, f(t), e, -f(t) * e))
        e = (1 - f(t)) * e
    return rows, e


# ------------------------- the 2-2-1 network (c) -------------------------
def hb(g, eta, gam, steps, p=P0):
    """Heavy ball with a schedule, v = gam v - eta_t g, as in Chapter 6."""
    p, v = p.copy(), np.zeros(len(p))
    hist = [loss_flat(p)]
    for t in range(steps):
        v = gam * v - eta(t) * g(p)
        p = p + v
        hist.append(loss_flat(p))
        if not np.isfinite(hist[-1]) or hist[-1] > 1e6:
            hist += [np.inf] * (steps - t - 1)
            break
    return np.array(hist)


def adam_s(g, eta, steps, p=P0, om1=OM1, om2=OM2, zeta=ZETA):
    """Adam of Chapter 14 with eta_t in place of eta."""
    p = p.copy()
    mt, r = np.zeros(len(p)), np.zeros(len(p))
    hist = [loss_flat(p)]
    for t in range(steps):
        gt = g(p)
        mt = om1 * mt + (1 - om1) * gt
        r = om2 * r + (1 - om2) * gt * gt
        mh = mt / (1 - om1 ** (t + 1))
        rh = r / (1 - om2 ** (t + 1))
        p = p - eta(t) * mh / (np.sqrt(rh) + zeta)
        hist.append(loss_flat(p))
        if not np.isfinite(hist[-1]) or hist[-1] > 1e6:
            hist += [np.inf] * (steps - t - 1)
            break
    return np.array(hist)


T_NET, TW = 2000, 100
NET_SET = [("HB", 0.1), ("HB", 1.5), ("HB", 3.0), ("Adam", 0.03),
           ("Adam", 1.0)]


def variants(e, T=T_NET, tw=TW):
    return {"constant": constant(e), "warm-up": warm(constant(e), tw, e),
            "cosine": cosine(e, T), "warm-up + cosine": cosine(e, T, tw=tw)}


def net_run(meth, f, b=None, T=T_NET):
    """Heavy ball (gamma = 0.9) or Adam on the full batch (b=None) or on
    mini-batches of b students."""
    if b is None:
        g = grad_flat
    else:
        gen = batches(b)
        g = lambda p: grad_flat(p, next(gen))
    return hb(g, f, 0.9, T) if meth == "HB" else adam_s(g, f, T)


def net_table(b=None, settings=NET_SET):
    return {(meth, e, k): net_run(meth, f, b)
            for meth, e in settings for k, f in variants(e).items()}


# ------------------------- the range test (d) -------------------------
def range_test(lo=1e-3, hi=10.0, N=300, b=2, gam=0.9, meth="HB"):
    """eta grows geometrically from lo to hi over N steps; one run.
    Returns the rates and the full-batch losses after each step."""
    etas = lo * (hi / lo) ** (np.arange(N) / (N - 1))
    gen = batches(b)
    g = lambda p: grad_flat(p, next(gen))
    f = lambda t: etas[t]
    h = (hb(g, f, gam, N) if meth == "HB" else adam_s(g, f, N))
    return etas, h[1:]


if __name__ == "__main__":
    lam, Q = np.linalg.eigh(Hc)
    print("== Centred linear model: eigenvalues of H, L*, sigma^2(w*)")
    print(f"lambda = {lam}  2/lambda_max = {2 / lam[-1]:.4f}"
          f"  L* = {Lstar:.4f}  sigma^2 = {spread(wc):.4f}")
    print(f"Var of a batch of two = sigma^2/2 * 6/7 = {SB2:.4f}")
    print(f"floor eta*Var/4 for eta=0.4: {0.4 * SB2 / 4:.4f},"
          f" eta=0.04: {0.04 * SB2 / 4:.4f}")
    print(f"starting gap L(0) - L* = {Llin(np.zeros(3)) - Lstar:.2f}")

    print("\n== Schedules, first values and the sum over 400 steps")
    for k, f in LIN_SCHED.items():
        S = sum(f(t) for t in range(T_LIN))
        print(f"{k:14s} eta_0..3 = " + " ".join(f"{f(t):.4f}"
                                                for t in range(4))
              + f"  eta_99 = {f(99):.4f}  eta_399 = {f(399):.6f}"
              f"  sum = {S:.2f}")

    print("\n== Predicted bias and variance parts, summed over the modes")
    for k, f in LIN_SCHED.items():
        r = []
        for T in (100, 400):
            l, S, bi, va = mode_split(f, T)
            r.append(f"T={T}: bias {bi.sum():.1e} var {va.sum():.2e}")
        print(f"{k:14s} " + "   ".join(r))
    print(f"mode noise variances s_j^2 = {mode_split(constant(0.1))[1]}")

    print("\n== Mini-batch SGD, k = 2, mean gap over 500 seeds")
    tab = lin_table(500)
    for k, (G, S, eL) in tab.items():
        print(f"{k:14s} gap(100) = {G[100]:.2e}  gap(200) = {G[200]:.2e}"
              f"  gap(400) = {G[400]:.2e}")

    print("\n== Hand example: cosine with warm-up, eta_max = 0.4,"
          " t_w = 4, T = 20, on the intercept mode")
    rows, e = hand()
    for t, eta, et, s in rows:
        print(f"t={t}  eta = {eta:.4f}  e_t = {et:.4f}  move = {s:.4f}")
    print(f"e_8 = {e:.4f}")

    print("\n== Symbolic examples: eta_0 = 0.4, q = 0.5, tau = 3")
    fs = {"step": step_decay(0.4, 0.5, 3), "exp": exponential(0.4, 0.5, 3),
          "1/t": inverse(0.4, 3)}
    for k, f in fs.items():
        print(f"{k:5s} " + " ".join(f"{f(t):.4f}" for t in range(9))
              + f"  sum = {sum(f(t) for t in range(9)):.4f}")
    print("one mode, lambda = 2, s^2 = 1:")
    for e in (0.1, 0.01):
        v = e / (2 * (2 - 2 * e))
        print(f"eta = {e}: v_inf = {v:.5f}  gap part = {v:.5f}"
              f"  bias factor (1 - eta lam)^20 = {(1 - 2 * e) ** 20:.5f}")
    f = sgdr(0.4, 4)
    print("SGDR, T_0 = 4, T_mult = 2: "
          + " ".join(f"{f(t):.4f}" for t in range(13)))
    print(f"Vaswani, d = 512, t_w = 4000: peak {512 ** -0.5 * 4000 ** -0.5:.3e}"
          f"  at s = 16000: {512 ** -0.5 * 16000 ** -0.5:.3e}")
    print(f"Goyal, k n = 8192: eta = 0.1 * 8192/256 = {0.1 * 8192 / 256:.1f}")

    print("\n== 2-2-1 network, Hessian at the start")
    ev = np.linalg.eigvalsh(hess_flat(P0))
    print(f"largest eigenvalue = {ev[-1]:.4f}  2/lambda_max = {2 / ev[-1]:.4f}"
          f"  2(1+gamma)/lambda_max = {3.8 / ev[-1]:.4f}")
    f = warm(constant(3.0), TW, 3.0)
    p, v = P0.copy(), np.zeros(9)
    for t in range(TW):
        v = 0.9 * v - f(t) * grad_flat(p)
        p = p + v
    lm = np.linalg.eigvalsh(hess_flat(p))[-1]
    print(f"heavy ball, warm-up to eta = 3 over {TW} steps: L = "
          f"{loss_flat(p):.4f}  largest eigenvalue = {lm:.4f}"
          f"  2(1+gamma)/lambda_max = {3.8 / lm:.4f}")

    print("\n== 2-2-1 network, full batch, 2000 steps:"
          " steps to linear L*, L(2000), largest L in steps 1-300")
    for (meth, e, k), h in net_table().items():
        print(f"{meth:4s} {e:<5g} {k:17s} {fmt(first(h, LIN))}  {h[-1]:.1e}"
              f"  {np.max(h[1:301]):.2e}")

    print("\n== Mini-batches of two, 2000 steps: steps to linear L*, L(2000)")
    mini = [("HB", 0.1), ("HB", 0.3), ("Adam", 0.03), ("Adam", 0.2)]
    for meth, e in mini:
        for k in ("constant", "cosine"):
            h = net_run(meth, variants(e)[k], b=2)
            print(f"{meth:4s} {e:<5g} {k:9s} {fmt(first(h, LIN))}  {h[-1]:.1e}"
                  f"  mean L(1001-2000) = {np.mean(h[1001:]):.1e}")

    print("\n== Adam, eta = 0.03, cosine, batches of two: the end point")
    gen = batches(2)
    p, mt, r = P0.copy(), np.zeros(9), np.zeros(9)
    f = cosine(0.03, T_NET)
    for t in range(T_NET):
        gt = grad_flat(p, next(gen))
        mt = OM1 * mt + (1 - OM1) * gt
        r = OM2 * r + (1 - OM2) * gt * gt
        p = p - f(t) * (mt / (1 - OM1 ** (t + 1))) / (
            np.sqrt(r / (1 - OM2 ** (t + 1))) + ZETA)
    print(f"L = {loss_flat(p):.3e}  Hessian eigenvalues "
          f"{np.linalg.eigvalsh(hess_flat(p))}")
    h = hb(grad_flat, constant(0.1), 0.9, 20000, p)
    print(f"full-batch heavy ball from there, eta = 0.1: L(20000) = "
          f"{h[-1]:.3e}")

    print("\n== Range test, mini-batches of two, 400 steps")
    for meth, lo, hi in (("HB", 1e-3, 10.0), ("Adam", 1e-4, 3.0)):
        e, h = range_test(lo, hi, 400, meth=meth)
        fin = np.isfinite(h)
        i = np.argmin(np.where(fin, h, np.inf))
        up = np.argmax(~fin | (h > loss_flat(P0)))
        dn = np.argmax(h < 0.9 * loss_flat(P0))
        print(f"{meth}: eta grows by {(hi / lo) ** (1 / 399):.5f} per step;"
              f" lowest L = {h[i]:.2e} at eta = {e[i]:.4f};"
              f" L above L(w_0) = {loss_flat(P0):.4f} from eta = {e[up]:.4f};"
              f" L below 0.9 L(w_0) from eta = {e[dn]:.4f}")
