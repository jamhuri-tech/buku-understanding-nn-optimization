"""Chapter 12: AdaGrad (Duchi, Hazan and Singer 2011)."""
import numpy as np
from students import X, y, H, w_star, loss, grad, m
from net221 import init, flat, loss_flat, grad_flat, train
from ch10_perweight import dbd, rprop, batches, first, LIN

np.set_printoptions(precision=4, suppress=True)
LS = loss(w_star)                 # linear L*, points^2
ZETA = 1e-8                       # the small constant zeta
C = 60.0                          # study time in minutes


def adagrad(g, L, w, eta, steps, zeta=ZETA, target=None, log=None):
    """Diagonal AdaGrad. Returns w, first hit, losses, history h."""
    w = w.copy()
    h = np.zeros(len(w))
    hist, hit = [L(w)], None
    for t in range(steps):
        gt = g(w)
        h = h + gt * gt
        rate = eta / (np.sqrt(h) + zeta)
        if log is not None:
            log.append((w.copy(), gt.copy(), h.copy(), rate.copy()))
        w = w - rate * gt
        hist.append(L(w))
        if hit is None and target is not None and hist[-1] <= target:
            hit = t + 1
    return w, hit, hist, h


def gd(g, L, w, eta, steps, target=None):
    """Plain gradient descent with the same outputs as adagrad()."""
    w = w.copy()
    hist, hit = [L(w)], None
    for t in range(steps):
        w = w - eta * g(w)
        hist.append(L(w))
        if hit is None and target is not None and hist[-1] <= target:
            hit = t + 1
    return w, hit, hist


def heavy_ball(g, L, w, eta, gam, steps):
    w, v = w.copy(), np.zeros(len(w))
    hist = [L(w)]
    for _ in range(steps):
        v = gam * v - eta * g(w)
        w = w + v
        hist.append(L(w))
    return w, None, hist


# Study time in minutes: column 1 of X times C.
XM = X.copy()
XM[:, 1] *= C
HM = XM.T @ XM / m


def loss_min(w):
    r = XM @ w - y
    return r @ r / (2 * m)


def grad_min(w):
    return XM.T @ (XM @ w - y) / m


def linear_runs(steps=200000):
    w0, T = np.zeros(3), LS + 0.01
    out = {"GD 0.03": gd(grad, loss, w0, 0.03, steps, T),
           "DBD": dbd(grad, loss, w0, 0.03, 0.01, 0.1, 0.7, steps, T),
           "Rprop+": rprop(grad, loss, w0, 0.1, steps, T)}
    for eta in (1, 2, 5, 10, 20):
        out[f"AdaGrad {eta}"] = adagrad(grad, loss, w0, eta, steps,
                                        target=T)
    return out


def minute_runs(steps=200000):
    w0, T = np.zeros(3), LS + 0.01
    ev = np.linalg.eigvalsh(HM)
    eta = 2 / (ev[0] + ev[-1])
    out = {"GD": gd(grad_min, loss_min, w0, eta, steps, T)}
    for e in (1, 10):
        out[f"AdaGrad {e}"] = adagrad(grad_min, loss_min, w0, e, steps,
                                      target=T)
    return out, ev, eta


def net_runs(steps=2000):
    p0 = flat(*init())
    return {"GD": train(0.1, steps)[2],
            "HB": train(0.1, steps, 0.9)[2],
            "Rprop+": rprop(grad_flat, loss_flat, p0, 0.1, steps)[2],
            "iRprop-": rprop(grad_flat, loss_flat, p0, 0.1, steps,
                             variant="-")[2],
            "AdaGrad 0.1": adagrad(grad_flat, loss_flat, p0, 0.1,
                                   steps)[2],
            "AdaGrad 0.5": adagrad(grad_flat, loss_flat, p0, 0.5,
                                   steps)[2]}


def minibatch_runs(b=2, steps=2000, eta=0.5):
    """SGD, heavy ball, Rprop+ and AdaGrad on the same mini-batches."""
    p0 = flat(*init())
    out = {}
    gen = batches(b)
    out["SGD"] = gd(lambda p: grad_flat(p, next(gen)), loss_flat, p0,
                    0.1, steps)[2]
    gen = batches(b)
    out["HB"] = heavy_ball(lambda p: grad_flat(p, next(gen)), loss_flat,
                           p0, 0.1, 0.9, steps)[2]
    gen = batches(b)
    out["Rprop+"] = rprop(lambda p: grad_flat(p, next(gen)), loss_flat,
                          p0, 0.1, steps)[2]
    gen = batches(b)
    log = []
    out["AdaGrad"] = adagrad(lambda p: grad_flat(p, next(gen)), loss_flat,
                             p0, eta, steps, log=log)[2]
    return out, log


def rate_log(eta=0.5, steps=2000, b=None):
    """Effective rates eta/(sqrt(h)+zeta) of the network, full or mini."""
    p0 = flat(*init())
    log = []
    if b is None:
        adagrad(grad_flat, loss_flat, p0, eta, steps, log=log)
    else:
        gen = batches(b)
        adagrad(lambda p: grad_flat(p, next(gen)), loss_flat, p0, eta,
                steps, log=log)
    return np.array([r for *_, r in log])


def fmt(k):
    return "  --" if k is None else f"{k:5d}"


if __name__ == "__main__":
    print("== AdaGrad, first two steps on the raw linear model, eta = 10")
    log = []
    adagrad(grad, loss, np.zeros(3), 10, 2, log=log)
    for t, (w, g, h, r) in enumerate(log):
        print(f"t={t} w = {w}  L = {loss(w):.4f}")
        print(f"   g = {g}")
        print(f"   h = {h}")
        print(f"   rate = {r}")
        print(f"   step = {-r * g}")
    w2 = log[1][0] - log[1][3] * log[1][1]
    print(f"t=2 w = {w2}  L = {loss(w2):.4f}")

    print("\n== Linear model, raw features, steps to L* + 0.01")
    print(f"L* = {LS:.4f}")
    for k, r in linear_runs().items():
        print(f"{k:12s} {fmt(r[1])}")

    print("\n== Study time in minutes")
    print("H =\n", HM)
    out, ev, eta = minute_runs()
    print(f"eigenvalues: {ev}  kappa = {ev[-1] / ev[0]:.4g}")
    print(f"2/lambda_max = {2 / ev[-1]:.3e}   GD eta = {eta:.3e}")
    gd_min = out.pop("GD")
    print(f"GD     {fmt(gd_min[1])}   gap at 200000 = "
          f"{gd_min[2][-1] - LS:.3f}")
    for k, r in out.items():
        print(f"{k:12s} {fmt(r[1])}   w = {r[0]}")
    g0 = grad_min(np.zeros(3))
    print("first gradient, minutes:", g0)

    print("\n== AdaGrad on the network, first two steps, eta = 0.5")
    log = []
    adagrad(grad_flat, loss_flat, flat(*init()), 0.5, 2, log=log)
    for t, (w, g, h, r) in enumerate(log):
        print(f"t={t}")
        print("  w    =", w)
        print("  g    =", g)
        print("  rate =", r)

    print("\n== 2-2-1 network, full batch, steps to linear L*")
    for k, h in net_runs().items():
        print(f"{k:12s} {fmt(first(h, LIN))} {fmt(first(h, 1e-4))}"
              f"   L(2000) = {h[-1]:.2e}")

    print("\n== Effective rates, full batch, eta = 0.5")
    R = rate_log()
    for t in (0, 9, 99, 1999):
        print(f"t={t + 1:5d} min = {R[t].min():.3f}  max = {R[t].max():.3f}")
    print("\n== Effective rates, mini-batches of two, eta = 0.5")
    R = rate_log(b=2)
    for t in (0, 9, 99, 1999):
        print(f"t={t + 1:5d} min = {R[t].min():.3f}  max = {R[t].max():.3f}")

    print("\n== Mini-batches of two, full-batch loss")
    out, _ = minibatch_runs()
    print("         " + "".join(f"{k:>10s}" for k in out))
    for t in (10, 50, 100, 500, 2000):
        print(f"t={t:5d}  " + "".join(f"{h[t]:10.2e}" for h in out.values()))
    print("to L*    " + "".join(f"{fmt(first(h, LIN)):>10s}"
                                for h in out.values()))
