"""Chapter 23: optimisers without schedules. Schedule-free SGD and
schedule-free AdamW (Defazio et al. 2024) on a noisy quadratic, the
linear model of the students and the 2-2-1 network."""
import numpy as np
from students import m
from net221 import init, flat, loss_flat, grad_flat
from ch05_stochastic import Xc, wc, L as Llin, Lstar, y
from ch10_perweight import batches, LIN, first
from ch14_adam import OM1, OM2, ZETA
from ch15_schedules import constant, cosine

np.set_printoptions(precision=4, suppress=True)
P0 = flat(*init())
MASK = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1.0])   # no decay on biases
BETA = 0.9                                        # interpolation beta'


def linear_decay(e0, T):
    """eta_t = e0 (1 - t/T), zero at t = T."""
    return lambda t: e0 * max(0.0, 1 - t / T)


# ------------------------- the sequences -------------------------------
def sf_coeffs(eta, h, beta, c):
    """One schedule-free step on L = h w^2 / 2 as s' = A s + b xi,
    with state s = (w, wbar) and gradient h u + xi at u."""
    a = 1 - eta * h * (1 - beta)
    A = np.array([[a, -eta * h * beta],
                  [c * a, 1 - c - c * eta * h * beta]])
    return A, np.array([-eta, -c * eta])


def quad(kind, T=1000, eta=0.5, h=1.0, sig=1.0, w0=10.0, beta=BETA,
         sched=None):
    """Exact mean and variance of the reported weight on the 1-D noisy
    quadratic. kind 'sgd' reports the last iterate with steps sched(t);
    kind 'sf' reports wbar. Returns the expected gaps E[h w^2 / 2]."""
    mean = np.array([w0, w0])
    C = np.zeros((2, 2))
    gaps = [h * w0 ** 2 / 2]
    for t in range(T):
        if kind == "sgd":
            e = sched(t)
            A, b = np.array([[1 - e * h, 0], [0, 1.0]]), np.array([-e, 0])
        else:
            A, b = sf_coeffs(eta, h, beta, 1 / (t + 2))
        mean = A @ mean
        C = A @ C @ A.T + sig ** 2 * np.outer(b, b)
        k = 0 if kind == "sgd" else 1
        gaps.append(h * (mean[k] ** 2 + C[k, k]) / 2)
    return np.array(gaps)


QUAD = {"constant": ("sgd", dict(sched=constant(0.5))),
        "linear decay": ("sgd", dict(sched=linear_decay(0.5, 1000))),
        "cosine": ("sgd", dict(sched=cosine(0.5, 1000))),
        "Polyak": ("sf", dict(beta=0.0)),
        "schedule-free": ("sf", dict(beta=BETA)),
        "primal": ("sf", dict(beta=1.0))}


def quad_table(ts=(10, 100, 1000)):
    out = {}
    for name, (kind, kw) in QUAD.items():
        g = quad(kind, **kw)
        out[name] = [g[t] for t in ts] + [1000 * g[1000]]
    return out


def noiseless_paths(T=12, eta=0.5, h=1.0, w0=10.0, beta=BETA):
    """w_t, wbar_t and u_t on L = h w^2/2 without noise."""
    w, wb, rows = w0, w0, []
    for t in range(T + 1):
        u = (1 - beta) * w + beta * wb
        rows.append((t, w, wb, u))
        w = w - eta * h * u
        c = 1 / (t + 2)
        wb = (1 - c) * wb + c * w
    return np.array(rows)


def primal_momentum(T=6):
    """Momentum t/(t+2) and step factor 1/(t+2) of primal averaging."""
    return [(t, t / (t + 2), 1 / (t + 2)) for t in range(1, T + 1)]


def adam_weights(tw=4, T=8, om2=None):
    """eta_t / eta and c_{t+1} = eta_t^2 / sum eta_i^2 with warm-up tw
    (om2 = None leaves out the bias factor sqrt(1 - om2^(t+1)))."""
    rows, S = [], 0.0
    for t in range(T):
        e = min(1.0, (t + 1) / tw)
        if om2 is not None:
            e *= np.sqrt(1 - om2 ** (t + 1))
        S += e * e
        rows.append((t, e, e * e / S))
    return rows


# ----------------------- the students: linear model --------------------
K = 2


def lin_run(kind, T, eta=1.0, beta=BETA, sched=None, seed=9):
    """Mini-batch steps on centred features, K distinct students drawn
    independently at each step. kind 'sgd' with steps sched(t), or 'sf'
    (beta = 0 is Polyak averaging). Returns the gaps of the reported
    weights for t = 0..T."""
    rng = np.random.default_rng(seed)
    w = np.zeros(3)
    wb = w.copy()
    gaps = [Llin(w) - Lstar]
    for t in range(T):
        B = rng.choice(m, K, replace=False)
        if kind == "sgd":
            w = w - sched(t) * Xc[B].T @ (Xc[B] @ w - y[B]) / K
            gaps.append(Llin(w) - Lstar)
        else:
            u = (1 - beta) * w + beta * wb
            w = w - eta * Xc[B].T @ (Xc[B] @ u - y[B]) / K
            c = 1 / (t + 2)
            wb = (1 - c) * wb + c * w
            gaps.append(Llin(wb) - Lstar)
    return np.array(gaps)


def mean_runs(f, n=500):
    return np.mean([f(s) for s in range(n)], axis=0)


def lin_hand(eta=1.0, beta=BETA, steps=3):
    """Full-batch schedule-free SGD from 0: rows t, u_t, g(u_t), w_{t+1},
    wbar_{t+1}."""
    w = np.zeros(3)
    wb = w.copy()
    rows = []
    for t in range(steps):
        u = (1 - beta) * w + beta * wb
        g = Xc.T @ (Xc @ u - y) / m
        w = w - eta * g
        c = 1 / (t + 2)
        wb = (1 - c) * wb + c * w
        rows.append((t, u, g, w, wb, Llin(wb) - Lstar))
    return rows


STOPS = (100, 200, 400, 800)


def lin_table(n=500):
    """Mean gaps at the stopping times. Scheduled runs restart for each
    stopping time T; schedule-free and constant runs are read off."""
    out = {}
    for name, e in (("constant 0.4", 0.4), ("constant 0.04", 0.04)):
        g = mean_runs(lambda s: lin_run("sgd", 800, sched=constant(e),
                                        seed=s), n)
        out[name] = [g[T] for T in STOPS]
    for name, f in (("cosine", cosine), ("linear decay", linear_decay)):
        out[name] = [mean_runs(lambda s: lin_run("sgd", T, sched=f(0.4, T),
                                                 seed=s), n)[-1]
                     for T in STOPS]
    for name, b, e in (("Polyak 0.4", 0.0, 0.4),
                       ("schedule-free", BETA, 1.0)):
        g = mean_runs(lambda s: lin_run("sf", 800, eta=e, beta=b, seed=s),
                      n)
        out[name] = [g[T] for T in STOPS]
    return out


def lin_eta(etas=(0.4, 1.0, 2.0, 4.0), n=500):
    return {e: mean_runs(lambda s: lin_run("sf", 800, eta=e, seed=s), n)
            for e in etas}


# ------------------------ the students: network -------------------------
def adamw(eta_f, T, rho=0.01, b=2, seed=9):
    """AdamW (Chapter 17) with steps eta_f(t) on batches of b."""
    gen = batches(b, seed)
    p = P0.copy()
    mo, r = 0 * p, 0 * p
    hist = [loss_flat(p)]
    for t in range(1, T + 1):
        g = grad_flat(p, next(gen))
        mo = OM1 * mo + (1 - OM1) * g
        r = OM2 * r + (1 - OM2) * g * g
        e = eta_f(t - 1)
        step = (mo / (1 - OM1 ** t)) / (np.sqrt(r / (1 - OM2 ** t)) + ZETA)
        p = p - e * rho * MASK * p - e * step
        hist.append(loss_flat(p))
    return np.array(hist)


def sf_adamw(eta, T, rho=0.01, beta=BETA, tw=100, b=2, seed=9, weight=2,
             log=None):
    """Schedule-free AdamW. weight=2 uses c = eta_t^2 / sum eta_i^2,
    weight=0 the plain average. Returns full-batch losses at wbar."""
    gen = batches(b, seed)
    w = P0.copy()
    wb, r = w.copy(), 0 * w
    hist, S = [loss_flat(wb)], 0.0
    for t in range(1, T + 1):
        u = (1 - beta) * w + beta * wb
        g = grad_flat(u, next(gen))
        r = OM2 * r + (1 - OM2) * g * g
        e = eta * np.sqrt(1 - OM2 ** t) * min(1.0, t / tw)
        w = w - e * g / (np.sqrt(r) + ZETA) - e * rho * MASK * u
        S += e ** weight
        c = e ** weight / S
        wb = (1 - c) * wb + c * w
        hist.append(loss_flat(wb))
        if log is not None:
            log.append((loss_flat(u), loss_flat(w)))
    return np.array(hist)


NET_STOPS = (500, 1000, 2000)


def net_table():
    rows = {}
    for e in (0.03, 0.1, 0.3):
        hs = [adamw(cosine(e, T, tw=100), T) for T in NET_STOPS]
        rows[("AdamW + cosine", e)] = ([h[-1] for h in hs],
                                       first(hs[-1], LIN))
    for e in (0.03, 0.1, 0.3):
        h = sf_adamw(e, 2000)
        rows[("schedule-free", e)] = ([h[T] for T in NET_STOPS],
                                      first(h, LIN))
    return rows


BETAS = (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 1.0)


def beta_table(n=500):
    rows = {}
    for b in BETAS:
        g = mean_runs(lambda s: lin_run("sf", 800, beta=b, seed=s), n)
        h = sf_adamw(0.1, 2000, beta=b)
        rows[b] = (g[200], g[800], h[500], h[2000], first(h, LIN))
    return rows


def fmt(k):
    return "never" if k is None else str(k)


def gfmt(x, w=10):
    return f"{'diverges':>{w}s}" if x > 1e6 else f"{x:{w}.4g}"


if __name__ == "__main__":
    print("== Noiseless 1-D quadratic, eta = 0.5, h = 1, w0 = 10")
    for t, w, wb, u in noiseless_paths(6):
        print(f"t={int(t)}  w={w:8.4f}  wbar={wb:8.4f}  u={u:8.4f}")

    print("\n== Primal averaging as momentum: t, t/(t+2), 1/(t+2)")
    for t, g, s in primal_momentum():
        print(f"t={t}  momentum={g:.4f}  step factor={s:.4f}")

    print("\n== Weights c_(t+1) with warm-up t_w = 4 (no bias factor)")
    for t, e, c in adam_weights():
        print(f"t={t}  eta_t/eta={e:.4f}  c={c:.4f}")
    print("== the same with the bias factor, omega_2 = 0.999")
    for t, e, c in adam_weights(om2=OM2):
        print(f"t={t}  eta_t/eta={e:.4f}  c={c:.4f}")

    print("\n== Noisy 1-D quadratic, h = 1, sigma = 1, eta = 0.5, w0 = 10")
    print("method          gap t=10   t=100     t=1000    1000*gap")
    for k, v in quad_table().items():
        print(f"{k:14s}" + "".join(f"{x:10.4g}" for x in v))
    print(f"bound sigma^2/(2hT) at T=1000: {1 / 2000:.4g}")

    print("\n== Students, linear model, full batch, eta = 1, beta' = 0.9")
    print(f"w* = {wc}, L* = {Lstar:.4f}")
    for t, u, g, w, wb, gap in lin_hand():
        print(f"t={t}  u={u}  g={g}")
        print(f"     w_{t + 1}={w}  wbar_{t + 1}={wb}  gap={gap:.2f}")

    print("\n== Students, linear model, batches of two, 500 runs")
    print("method          gap T=100  T=200     T=400     T=800")
    for k, v in lin_table().items():
        print(f"{k:14s}" + "".join(gfmt(x) for x in v))
    g = mean_runs(lambda s: lin_run("sgd", 800, sched=cosine(0.4, 800),
                                    seed=s))
    print(f"cosine planned for 800, read at t=100: {g[100]:.4g}  "
          f"t=200: {g[200]:.4g}")
    print("schedule-free, other eta (gaps at 100, 200, 400, 800)")
    for e, g in lin_eta().items():
        print(f"eta={e}: " + "".join(gfmt(g[T]) for T in STOPS))

    print(f"\n== Network, batches of two, rho = 0.01, t_w = 100; "
          f"L*_lin = {LIN:.6f}")
    print("method            eta    T=500     T=1000    T=2000  to L*")
    for (k, e), (v, s) in net_table().items():
        print(f"{k:16s}{e:5.2f}" + "".join(f"{x:10.3g}" for x in v)
              + f"  {fmt(s)}")
    h0 = sf_adamw(0.1, 2000, weight=0)
    print(f"schedule-free 0.1, plain average: T=500 {h0[500]:.3g}  "
          f"T=2000 {h0[2000]:.3g}  to L* {fmt(first(h0, LIN))}")
    log = []
    sf_adamw(0.1, 2000, log=log)
    print(f"schedule-free 0.1 at t=2000: L(u) {log[-1][0]:.3g}  "
          f"L(w) {log[-1][1]:.3g}")

    print("\n== Sensitivity to beta' (linear eta = 1; net eta = 0.1)")
    print("beta'  lin T=200   lin T=800   net T=500  net T=2000  to L*")
    for b, (g2, g8, h5, h20, s) in beta_table().items():
        print(f"{b:5.2f}{gfmt(g2, 11)}{gfmt(g8, 12)}{h5:11.3g}{h20:12.3g}"
              f"  {fmt(s)}")
