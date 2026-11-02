"""Chapter 10: per-weight learning rates, delta-bar-delta and Rprop."""
import numpy as np
from students import X, y, H, w_star, loss, grad
from net221 import init, flat, loss_flat, grad_flat, train, SY

np.set_printoptions(precision=4, suppress=True)
LS = loss(w_star)                 # linear L*, points^2
LIN = 0.289198 / SY ** 2          # linear L* in standard units


def dbd(g, L, w, eta0, a, phi, theta, steps, target=None):
    """Delta-bar-delta (Jacobs 1988). Returns w, first hit, losses, rates."""
    w = w.copy()
    eta = np.full(len(w), eta0)
    gbar = np.zeros(len(w))
    hist, hit = [L(w)], None
    for t in range(steps):
        gt = g(w)
        s = gbar * gt
        eta = np.where(s > 0, eta + a, np.where(s < 0, (1 - phi) * eta, eta))
        w = w - eta * gt
        gbar = (1 - theta) * gt + theta * gbar
        hist.append(L(w))
        if hit is None and target is not None and hist[-1] <= target:
            hit = t + 1
    return w, hit, hist, eta


def rprop(g, L, w, d0, steps, target=None, up=1.2, down=0.5,
          dmax=50.0, dmin=1e-6, variant="+", log=None):
    """Rprop+ (weight backtracking) or iRprop- (variant "-")."""
    w = w.copy()
    D = np.full(len(w), d0)
    gp, dw = np.zeros(len(w)), np.zeros(len(w))
    hist, hit = [L(w)], None
    for t in range(steps):
        gt = g(w)
        s = gp * gt
        D = np.where(s > 0, np.minimum(up * D, dmax),
                     np.where(s < 0, np.maximum(down * D, dmin), D))
        if variant == "+":
            new = np.where(s < 0, -dw, -np.sign(gt) * D)
            gp = np.where(s < 0, 0.0, gt)
        else:
            gt = np.where(s < 0, 0.0, gt)
            new = -np.sign(gt) * D
            gp = gt
        if log is not None:
            log.append((w.copy(), gt.copy(), D.copy(), new.copy()))
        w = w + new
        dw = new
        hist.append(L(w))
        if hit is None and target is not None and hist[-1] <= target:
            hit = t + 1
    return w, hit, hist, D


def batches(b, seed=9):
    """Mini-batches of size b; a fresh shuffle every epoch."""
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(len(y))
        for k in range(0, len(y), b):
            yield perm[k:k + b]


def linear_runs(steps=20000):
    w0, T = np.zeros(3), LS + 0.01
    ev = np.linalg.eigvalsh(H)
    eta = 2 / (ev[0] + ev[-1])
    hist = [loss(w0)]
    w = w0.copy()
    for _ in range(steps):
        w = w - eta * grad(w)
        hist.append(loss(w))
    hit = first(hist, T)
    Dj = 1 / np.diag(H)
    ej = np.linalg.eigvalsh(np.diag(Dj ** 0.5) @ H @ np.diag(Dj ** 0.5))
    ej = 2 / (ej[0] + ej[-1])
    hj, w = [loss(w0)], w0.copy()
    for _ in range(steps):
        w = w - ej * Dj * grad(w)
        hj.append(loss(w))
    return {"GD": (hit, hist, eta), "Jacobi": (first(hj, T), hj, ej),
            "DBD": dbd(grad, loss, w0, 0.03, 0.01, 0.1, 0.7, steps, T),
            "Rprop+": rprop(grad, loss, w0, 0.1, steps, T),
            "iRprop-": rprop(grad, loss, w0, 0.1, steps, T, variant="-")}


def net_runs(steps=2000):
    p0 = flat(*init())
    return {"GD": train(0.1, steps)[2],
            "HB": train(0.1, steps, 0.9)[2],
            "NAG": train(0.1, steps, 0.9, nesterov=True)[2],
            "DBD": dbd(grad_flat, loss_flat, p0, 0.1, 0.01, 0.1, 0.7,
                       steps)[2],
            "Rprop+": rprop(grad_flat, loss_flat, p0, 0.1, steps)[2],
            "iRprop-": rprop(grad_flat, loss_flat, p0, 0.1, steps,
                             variant="-")[2]}


def minibatch_runs(b=2, steps=2000):
    p0 = flat(*init())
    gen = batches(b)
    log = []
    rp = rprop(lambda p: grad_flat(p, next(gen)), loss_flat, p0, 0.1,
               steps, log=log)
    gen = batches(b)
    sgd = dbd(lambda p: grad_flat(p, next(gen)), loss_flat, p0, 0.1, 0, 0,
              0, steps)
    return rp, sgd, log


def first(h, level):
    return next((t for t, v in enumerate(h) if v <= level), None)


if __name__ == "__main__":
    print("== Diagonal scaling of H (raw features)")
    print("H =\n", H)
    ev = np.linalg.eigvalsh(H)
    print("eigenvalues:", ev.round(4), " kappa =", round(ev[-1] / ev[0], 1))
    S = np.diag(np.diag(H) ** -0.5)
    J = S @ H @ S
    ej = np.linalg.eigvalsh(J)
    print("diag(H)^(-1/2) =", np.diag(S).round(4))
    print("Jacobi-scaled H =\n", J)
    print("eigenvalues:", ej.round(4), " kappa =", round(ej[-1] / ej[0], 1))
    from scipy.optimize import minimize
    kap = lambda q: (lambda e: e[-1] / e[0])(
        np.linalg.eigvalsh(np.diag(np.exp(q)) @ H @ np.diag(np.exp(q))))
    r = minimize(kap, np.log(np.diag(S)), method="Nelder-Mead",
                 options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20000})
    print("best diagonal: kappa =", round(r.fun, 1))

    print("\n== Delta-bar-delta, first steps on the raw linear model")
    w, eta, gbar = np.zeros(3), np.full(3, 0.03), np.zeros(3)
    for t in range(3):
        gt = grad(w)
        s = gbar * gt
        eta = np.where(s > 0, eta + 0.01, np.where(s < 0, 0.9 * eta, eta))
        w = w - eta * gt
        print(f"t={t} g =", gt.round(4), "eta =", eta.round(4))
        print("     w =", w.round(4), " L =", round(loss(w), 4))
        gbar = 0.3 * gt + 0.7 * gbar
        print("  gbar =", gbar.round(4))

    print("\n== Linear model, raw features, steps to L* + 0.01")
    runs = linear_runs()
    print(f"GD eta = 2/(mu+beta) = {runs['GD'][2]:.5f}: {runs['GD'][0]}")
    print(f"Jacobi GD, eta = {runs['Jacobi'][2]:.4f}: {runs['Jacobi'][0]}")
    for k in ("DBD", "Rprop+", "iRprop-"):
        print(f"{k:8s}{runs[k][1]:6d}   final rates:", runs[k][3].round(4))
    print("Rprop+ L(1000) - L* =", f"{runs['Rprop+'][2][1000] - LS:.2e}")

    print("\n== Rprop+ on the network, first three steps")
    log = []
    rprop(grad_flat, loss_flat, flat(*init()), 0.1, 3, log=log)
    for t, (w, g, D, dw) in enumerate(log):
        print(f"t={t}")
        print("  w  =", w.round(4))
        print("  g  =", g.round(4))
        print("  D  =", D.round(4))
        print("  dw =", dw.round(4))

    print("\n== 2-2-1 network, full batch, steps to linear L*")
    nr = net_runs()
    for k, h in nr.items():
        hit = [str(first(h, v) or "--") for v in (LIN, 1e-4)]
        print(f"{k:8s}{hit[0]:>5s}{hit[1]:>6s}   L(2000) = {h[-1]:.2e}")

    print("\n== Signs of the per-student intercept gradients at w*")
    gi = X[:, 0] * (X @ w_star - y)
    print("x_i'w* - y_i =", gi.round(3), " sum =", round(gi.sum(), 10))
    print("positive:", int((gi > 0).sum()), " negative:", int((gi < 0).sum()))

    print("\n== Mini-batches of 2, Rprop+ and SGD (eta = 0.1)")
    rp, sgd, log = minibatch_runs()
    for t in (10, 50, 100, 500, 2000):
        print(f"t={t:5d}  Rprop+ L = {rp[2][t]:.4f}  max D = "
              f"{log[t - 1][2].max():.1e}   SGD L = {sgd[2][t]:.2e}")
    print("SGD steps to linear L*:", first(sgd[2], LIN),
          "  Rprop+:", first(rp[2], LIN))
    full = []
    fb = rprop(grad_flat, loss_flat, flat(*init()), 0.1, 100, log=full)
    for t in (50, 100):
        print(f"full batch t={t}: L = {fb[2][t]:.2e}  max D = "
              f"{full[t - 1][2].max():.1e}")
