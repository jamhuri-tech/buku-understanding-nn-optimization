"""Chapter 11: mini-batch SGD, Polyak-Ruppert averaging, and the shape
of the network loss."""
import numpy as np
from students import m
from net221 import (init, flat, unflat, forward, loss_flat, grad_flat,
                    sample_grads, jacobian, hess_flat, Xs, ys, SY)
from ch05_stochastic import Xc, Hc, wc, L as Llin, Lstar, y
from ch10_perweight import batches

np.set_printoptions(precision=4, suppress=True)
P0 = flat(*init())
LIN = 0.289198 / SY ** 2          # linear L* in standard units


def noise(p):
    """Trace of the noise covariance, sigma^2(w), Equation (5.variance)."""
    G = sample_grads(p)
    return np.mean(np.sum((G - G.mean(0)) ** 2, axis=1))


def sgd(k, eta, T, p=None, seed=9, tail=None):
    """Shuffled mini-batches of k students (k = 8 is gradient descent).

    Returns the last weights, the losses L(w_t), and the average of the
    iterates w_t for t > T - tail (tail=None means no average)."""
    p = P0.copy() if p is None else p.copy()
    g = batches(k, seed)
    hist, s, n = [loss_flat(p)], np.zeros_like(p), 0
    for t in range(T):
        p = p - eta * grad_flat(p, next(g))
        hist.append(loss_flat(p))
        if tail is not None and t >= T - tail:
            s, n = s + p, n + 1
    return p, np.array(hist), (s / n if n else None)


# ---------- the linear model: Polyak-Ruppert on centred features ----------
def lin_runs(T, seed, steps):
    """LMS with replacement from 0. Returns the gap of the last iterate
    and of the average over the second half, for each schedule."""
    out = []
    for eta in steps:
        rng = np.random.default_rng(seed)
        w, s = np.zeros(3), np.zeros(3)
        for t in range(T):
            i = rng.integers(m)
            w = w - eta(t) * (Xc[i] @ w - y[i]) * Xc[i]
            if t >= T // 2:
                s += w
        out.append((Llin(w) - Lstar, Llin(s / (T - T // 2)) - Lstar))
    return out


STEPS = {"const": lambda t: 0.1,
         "decay": lambda t: 0.2 * (t + 1) ** -0.6}


def lin_curve(T=80000, seeds=20):
    """Mean gaps of the last iterate (eta = 0.1) and of the half-tail
    average over [t/2, t], at log-spaced t."""
    ts = np.unique(np.round(np.logspace(1, np.log10(T), 40)).astype(int))
    last, avg = np.zeros(len(ts)), np.zeros(len(ts))
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        w, W = np.zeros(3), [np.zeros(3)]
        for t in range(T):
            i = rng.integers(m)
            w = w - 0.1 * (Xc[i] @ w - y[i]) * Xc[i]
            W.append(w)
        C = np.cumsum(np.array(W), axis=0)
        for j, t in enumerate(ts):
            last[j] += Llin(W[t]) - Lstar
            avg[j] += Llin((C[t] - C[t // 2]) / (t - t // 2)) - Lstar
    return ts, last / seeds, avg / seeds


def section_linear():
    print("== linear model, centred features, k = 1 with replacement")
    G = (Xc @ wc - y)[:, None] * Xc
    S = G.T @ G / m
    tr = np.trace(np.linalg.solve(Hc, S))
    print("H =\n", Hc)
    print("H^-1 =\n", np.linalg.inv(Hc))
    print("Sigma at w* =\n", S)
    print(f"sigma^2 = tr Sigma = {np.trace(S):.4f}   "
          f"tr(H^-1 Sigma) = {tr:.4f}")
    print(" T      last 0.1   tail 0.1   last dec   tail dec   tr/T")
    for T in (800, 8000, 80000):
        r = np.mean([lin_runs(T, s, STEPS.values()) for s in range(20)],
                    axis=0)
        print(f"{T:6d}  {r[0, 0]:.2e}   {r[0, 1]:.2e}   {r[1, 0]:.2e}"
              f"   {r[1, 1]:.2e}   {tr / T:.2e}")
    full = np.mean([lin_runs(8000, s, [STEPS["const"]])[0][0]
                    for s in range(20)])
    rng_full = []
    for s in range(20):
        rng = np.random.default_rng(s)
        w, acc = np.zeros(3), np.zeros(3)
        for t in range(8000):
            i = rng.integers(m)
            w = w - 0.1 * (Xc[i] @ w - y[i]) * Xc[i]
            acc += w
        rng_full.append(Llin(acc / 8000) - Lstar)
    print(f"T = 8000, eta = 0.1: last {full:.2e}, "
          f"average of all iterates {np.mean(rng_full):.2e}")


# ---------- the network ----------
def section_handbatch():
    print("== one mini-batch of two at w0")
    B = next(batches(2))
    W1, w2 = unflat(P0)
    _, At, yh = forward(W1, w2)
    e = yh - ys
    print("batch", B + 1, " errors", e[B], " a~_i:")
    print(At[B])
    gB = grad_flat(P0, B)
    g = grad_flat(P0)
    print("g_B (w2 part)", gB[6:], "  full (w2 part)", g[6:])
    print("g_B (W1 part)", gB[:6])
    s2 = noise(P0)
    print(f"|g_B - g| = {np.linalg.norm(gB - g):.4f}   |g| = "
          f"{np.linalg.norm(g):.4f}")
    print(f"sigma^2(w0) = {s2:.4f}   batch of 2 without replacement: "
          f"{s2 / 2 * 6 / 7:.4f}   with: {s2 / 2:.4f}")
    rng = np.random.default_rng(9)
    d = []
    for _ in range(20000):
        idx = rng.choice(m, 2, replace=False)
        d.append(np.sum((grad_flat(P0, idx) - g) ** 2))
    print(f"sampled mean |g_B - g|^2 over 20000 batches = {np.mean(d):.4f}")


def section_batches():
    print("== mini-batch SGD on the net, eta = 0.1")
    print("k   L(50)      L(200)     L(2000)    sigma^2/L at 2000")
    for k in (1, 2, 4, 8):
        p, h, _ = sgd(k, 0.1, 2000)
        print(f"{k}   {h[50]:.2e}   {h[200]:.2e}   {h[2000]:.2e}   "
              f"{noise(p) / h[2000]:.2f}")
    for k in (1, 2, 4, 8):
        h = sgd(k, 0.1, 2000)[1]
        print(f"k = {k}: first step below linear L* = {np.argmax(h < LIN)}")
    p, h, _ = sgd(1, 0.1, 2000)
    J = jacobian(p)
    print(f"k = 1 at t = 2000: 2 max|grad f_i|^2 = "
          f"{2 * np.max(np.sum(J ** 2, axis=1)):.2f}")


def noise_path(k=1, T=2000):
    """Loss and sigma^2 along an SGD run, every 10 steps."""
    p, g = P0.copy(), batches(k)
    L, S = [], []
    for t in range(T + 1):
        if t % 10 == 0:
            L.append(loss_flat(p))
            S.append(noise(p))
        p = p - 0.1 * grad_flat(p, next(g))
    return np.array(L), np.array(S)


def section_average_net():
    print("== Polyak-Ruppert on the net, k = 1, eta = 0.1")
    for T in (200, 2000, 20000):
        p, h, a = sgd(1, 0.1, T, tail=T // 2)
        print(f"T = {T:5d}   last {h[-1]:.2e}   tail {loss_flat(a):.2e}"
              f"   |avg - last| = {np.linalg.norm(a - p):.4f}")


def section_plateau():
    print("== plateaus near the saddle at zero, gradient descent eta = 0.1")
    for s in (1.0, 0.1, 0.01, 0.001):
        _, h, _ = sgd(8, 0.1, 400, p=s * P0)
        _, h1, _ = sgd(1, 0.1, 400, p=s * P0)
        print(f"scale {s:g}: steps to L < 0.1  GD {np.argmax(h < 0.1)}"
              f"  SGD k=1 {np.argmax(h1 < 0.1)}")
    print(f"predicted steps per factor 10: "
          f"{np.log(10) / np.log(1 + 0.1 * 1.1835):.1f}")
    p, _, _ = sgd(1, 0.1, 200, p=np.zeros(9))
    print("SGD k=1 from zero, 200 steps:", p)


def section_hessian():
    print("== Hessian eigenvalues")
    pg, h, _ = sgd(8, 0.1, 20000)
    for name, p in (("w0", P0), ("zero", np.zeros(9)), ("GD min", pg)):
        ev = np.linalg.eigvalsh(hess_flat(p))
        print(f"{name:7s} L = {loss_flat(p):.2e}  eig:", ev)
    print("GD min weights:", pg)
    p, neg = P0.copy(), []
    for t in range(1001):
        if t % 10 == 0:
            neg.append((t, int(np.sum(np.linalg.eigvalsh(hess_flat(p))
                                      < -1e-6))))
        p = p - 0.1 * grad_flat(p)
    print("negative eigenvalues along GD (t, count):",
          [n for n in neg if n[0] in (0, 10, 40, 1000)])
    print(f"smallest eigenvalue at t = 1000: "
          f"{np.linalg.eigvalsh(hess_flat(p))[0]:.2e}")
    sh = [np.linalg.eigvalsh(hess_flat(sgd(k, 0.1, 20000)[0]))[-1]
          for k in (1, 2, 4)]
    print("eta = 0.1, 20000 steps, lam_max for k = 1, 2, 4:",
          np.round(sh, 4))
    print(f"rank of J at GD min: {np.linalg.matrix_rank(jacobian(pg))}")


def epoch_growth(p, eta=0.5, n=200, seed=0):
    """Mean spectral radius of one epoch of linearised single-student
    steps, the product of (I - eta J_i J_i') in a random order."""
    J, rng, rs = jacobian(p), np.random.default_rng(seed), []
    for _ in range(n):
        M = np.eye(len(p))
        for i in rng.permutation(m):
            M = (np.eye(len(p)) - eta * np.outer(J[i], J[i])) @ M
        rs.append(np.max(np.abs(np.linalg.eigvals(M))))
    return np.mean(rs)


def section_sharp():
    print("== sharpness after 5000 steps, eta = 0.5")
    print("k   L          lam_max   tr H     mean lam_max")
    ends, rng_k = {}, []
    for k in (1, 2, 4, 8):
        p, h, _ = sgd(k, 0.5, 5000)
        ends[k] = p
        ev = np.linalg.eigvalsh(hess_flat(p))
        seeds = [np.linalg.eigvalsh(hess_flat(
            sgd(k, 0.5, 5000, seed=s)[0]))[-1] for s in range(10)]
        mean = np.mean(seeds)
        rng_k.append((k, min(seeds), max(seeds)))
        print(f"{k}   {h[-1]:.2e}   {ev[-1]:.4f}    {ev.sum():.4f}   "
              f"{mean:.2f}")
    for k, lo, hi in rng_k:
        print(f"k = {k}: ten seeds between {lo:.2f} and {hi:.2f}")
    print("SGD k=1 min:", ends[1])
    print("GD      min:", ends[8])
    print(f"distance {np.linalg.norm(ends[1] - ends[8]):.4f}")
    print(f"epoch growth, eta = 0.5: at GD min {epoch_growth(ends[8]):.2f}"
          f"  at SGD k=1 min {epoch_growth(ends[1]):.2f}")
    mid = loss_flat((ends[1] + ends[8]) / 2)
    print(f"loss at the midpoint {mid:.2e}")
    return ends


def slices(ends, r=np.linspace(-1, 1, 201)):
    """Loss along the top and the flattest eigenvector at two minima."""
    out = {}
    for k in (1, 8):
        p = ends[k]
        ev, V = np.linalg.eigh(hess_flat(p))
        out[k] = ([loss_flat(p + s * V[:, -1]) for s in r],
                  [loss_flat(p + s * V[:, 0]) for s in r], ev)
    return r, out


if __name__ == "__main__":
    section_linear()
    section_handbatch()
    section_batches()
    section_average_net()
    section_plateau()
    section_hessian()
    ends = section_sharp()
    r, out = slices(ends, np.array([-0.5, 0.5]))
    for k in (1, 8):
        print(f"k = {k}: L at +-0.5 along top {np.round(out[k][0], 4)}"
              f"  along flattest {np.array(out[k][1])}")
