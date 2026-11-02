"""Chapter 13: RMSProp (Tieleman and Hinton 2012), AdaDelta (Zeiler 2012)."""
import numpy as np
from students import H, w_star, loss, grad
from net221 import init, flat, loss_flat, grad_flat, train
from ch10_perweight import rprop, batches, first, LIN
from ch12_adagrad import adagrad, gd, heavy_ball, fmt

np.set_printoptions(precision=4, suppress=True)
LS = loss(w_star)                 # linear L*, points^2
ZETA = 1e-8                       # small constant of RMSProp
OMEGA = 0.9                       # decay of RMSProp
OM_AD, Z_AD = 0.95, 1e-6          # AdaDelta settings of Zeiler (2012)


def rmsprop(g, L, w, eta, omega=OMEGA, steps=1, zeta=ZETA, target=None,
            log=None, gam=0.0):
    """RMSProp; gam > 0 adds heavy-ball momentum to the scaled step."""
    w = w.copy()
    r, v = np.zeros(len(w)), np.zeros(len(w))
    hist, hit = [L(w)], None
    for t in range(steps):
        gt = g(w)
        r = omega * r + (1 - omega) * gt * gt
        rate = eta / (np.sqrt(r) + zeta)
        if log is not None:
            log.append((w.copy(), gt.copy(), r.copy(), rate.copy()))
        v = gam * v - rate * gt
        w = w + v
        hist.append(L(w))
        if hit is None and target is not None and hist[-1] <= target:
            hit = t + 1
    return w, hit, hist, r


def adadelta(g, L, w, omega=OM_AD, steps=1, zeta=Z_AD, target=None,
             log=None):
    """AdaDelta: step s = -sqrt(u + zeta)/sqrt(r + zeta) * g."""
    w = w.copy()
    r, u = np.zeros(len(w)), np.zeros(len(w))
    hist, hit = [L(w)], None
    for t in range(steps):
        gt = g(w)
        r = omega * r + (1 - omega) * gt * gt
        rate = np.sqrt(u + zeta) / np.sqrt(r + zeta)
        s = -rate * gt
        if log is not None:
            log.append((w.copy(), gt.copy(), r.copy(), u.copy(), rate.copy(),
                        s.copy()))
        u = omega * u + (1 - omega) * s * s
        w = w + s
        hist.append(L(w))
        if hit is None and target is not None and hist[-1] <= target:
            hit = t + 1
    return w, hit, hist, (r, u)


def bias_table(omega=OMEGA, ts=(0, 1, 2, 4, 9, 19, 49)):
    """Constant gradient: r_t = (1 - omega^(t+1)) g^2, move = eta/sqrt(.)."""
    return [(t, 1 - omega ** (t + 1), 1 / np.sqrt(1 - omega ** (t + 1)))
            for t in ts]


def cycle_1d(lam=2.0, eta=0.1, w0=1.0, steps=400):
    """RMSProp on L = lam w^2 / 2; returns the last iterates."""
    g = lambda w: lam * w
    L = lambda w: lam * w[0] ** 2 / 2
    log = []
    rmsprop(g, L, np.array([w0]), eta, steps=steps, log=log)
    return [x[0][0] for x in log[-4:]], log[-1][2][0]


def linear_runs(steps=20000):
    w0, T = np.zeros(3), LS + 0.01
    out = {"GD 0.03": gd(grad, loss, w0, 0.03, steps, T),
           "AdaGrad 10": adagrad(grad, loss, w0, 10, steps, target=T)[:3]}
    for eta in (0.003, 0.01, 0.03):
        out[f"RMSProp {eta}"] = rmsprop(grad, loss, w0, eta, steps=steps,
                                        target=T)[:3]
    out["AdaDelta"] = adadelta(grad, loss, w0, steps=steps, target=T)[:3]
    return out


def linear_rates(steps=20000):
    """Effective rate of w_2 for AdaGrad 10 and RMSProp 0.01."""
    la, lr = [], []
    adagrad(grad, loss, np.zeros(3), 10, steps, log=la)
    rmsprop(grad, loss, np.zeros(3), 0.01, steps=steps, log=lr)
    return (np.array([x[3][2] for x in la]), np.array([x[3][2] for x in lr]))


def net_runs(steps=2000):
    p0 = flat(*init())
    return {"GD": train(0.1, steps)[2],
            "HB": train(0.1, steps, 0.9)[2],
            "Rprop+": rprop(grad_flat, loss_flat, p0, 0.1, steps)[2],
            "AdaGrad 0.01": adagrad(grad_flat, loss_flat, p0, 0.01,
                                    steps)[2],
            "AdaGrad 0.5": adagrad(grad_flat, loss_flat, p0, 0.5, steps)[2],
            "RMSProp 0.01": rmsprop(grad_flat, loss_flat, p0, 0.01,
                                    steps=steps)[2],
            "AdaDelta": adadelta(grad_flat, loss_flat, p0, steps=steps)[2]}


def minibatch_runs(b=2, steps=2000):
    """Six methods on the same mini-batches of b students."""
    p0 = flat(*init())
    runs = {"SGD": lambda g: gd(g, loss_flat, p0, 0.1, steps),
            "HB": lambda g: heavy_ball(g, loss_flat, p0, 0.1, 0.9, steps),
            "Rprop+": lambda g: rprop(g, loss_flat, p0, 0.1, steps),
            "AdaGrad": lambda g: adagrad(g, loss_flat, p0, 0.5, steps),
            "RMSProp": lambda g: rmsprop(g, loss_flat, p0, 0.01,
                                         steps=steps),
            "AdaDelta": lambda g: adadelta(g, loss_flat, p0, steps=steps)}
    out = {}
    for k, run in runs.items():
        gen = batches(b)
        out[k] = run(lambda p: grad_flat(p, next(gen)))[2]
    return out


def net_rates(eta=0.01, steps=2000, b=2):
    """Effective rates of AdaGrad and RMSProp, same eta, mini-batches."""
    p0 = flat(*init())
    la, lr = [], []
    gen = batches(b)
    adagrad(lambda p: grad_flat(p, next(gen)), loss_flat, p0, eta, steps,
            log=la)
    gen = batches(b)
    rmsprop(lambda p: grad_flat(p, next(gen)), loss_flat, p0, eta,
            steps=steps, log=lr)
    return (np.array([x[3] for x in la]), np.array([x[3] for x in lr]),
            loss_flat(la[-1][0]), loss_flat(lr[-1][0]))


def sensitivity(steps=2000, eta=0.01):
    """Full-batch net: steps to linear L* and mean loss over 1001-2000."""
    p0 = flat(*init())
    out = {}
    for om in (0.5, 0.9, 0.99, 0.999):
        for z in (1e-8, 1e-4, 1e-2, 1e-1):
            h = rmsprop(grad_flat, loss_flat, p0, eta, om, steps, z)[2]
            out[(om, z)] = (first(h, LIN), np.mean(h[1001:]))
    return out


if __name__ == "__main__":
    print("== Constant gradient, omega = 0.9: r_t/g^2 and move/eta")
    for t, f, s in bias_table():
        print(f"t={t:3d}  r/g^2 = {f:.4f}  move/eta = {s:.4f}")

    print("\n== RMSProp, first two steps on the raw linear model,"
          " eta = 0.01, omega = 0.9")
    log = []
    rmsprop(grad, loss, np.zeros(3), 0.01, steps=2, log=log)
    for t, (w, g, r, rt) in enumerate(log):
        print(f"t={t} w = {w}  L = {loss(w):.4f}")
        print(f"   g = {g}")
        print(f"   r = {r}")
        print(f"   rate = {rt * 1e4} x 1e-4")
        print(f"   move = {-rt * g}")
    w2 = log[1][0] - log[1][3] * log[1][1]
    print(f"t=2 w = {w2}  L = {loss(w2):.4f}")

    print("\n== AdaDelta, first two steps on the raw linear model,"
          " omega = 0.95, zeta = 1e-6")
    log = []
    adadelta(grad, loss, np.zeros(3), steps=2, log=log)
    for t, (w, g, r, u, rt, s) in enumerate(log):
        print(f"t={t} w = {w}")
        print(f"   g = {g}")
        print(f"   r = {r}")
        print(f"   u_prev = {u[0]:.4e} (all three)")
        print(f"   rate = {rt * 1e5} x 1e-5")
        print(f"   move = {s[0]:.6f} {s[1]:.6f} {s[2]:.6f}")

    print("\n== 1-D quadratic, lam = 2, eta = 0.1: iterates after 400 steps")
    for lam in (2.0, 50.0):
        ws, r = cycle_1d(lam=lam)
        print(f"lam = {lam}: w = {np.round(ws, 6)}  r = {r:.6f}")

    print("\n== Linear model: predicted RMSProp floor eta^2/8 * 1'H1")
    s = np.ones(3) @ H @ np.ones(3)
    print(f"1'H1 = {s:.4f}")
    out = linear_runs()
    print("\n== Linear model, raw features, steps to L* + 0.01")
    print(f"L* = {LS:.4f}")
    for k, r in out.items():
        h = r[2]
        print(f"{k:13s} {fmt(r[1])}   gap(20000) = {h[-1] - LS:.3e}")
    for eta in (0.003, 0.01, 0.03):
        print(f"predicted floor, eta = {eta}: {eta ** 2 / 8 * s:.3e}")
    h = out["AdaDelta"][2]
    print("AdaDelta gaps at 10, 100, 1000, 5000, 20000:",
          " ".join(f"{h[t] - LS:.3g}" for t in (10, 100, 1000, 5000, 20000)))

    ra, rr = linear_rates()
    print("\n== Effective rate of w2, linear model")
    for t in (0, 9, 99, 999, 19999):
        print(f"t={t + 1:6d}  AdaGrad {ra[t]:.3e}  RMSProp {rr[t]:.3e}")

    print("\n== 2-2-1 network, full batch")
    for k, h in net_runs().items():
        print(f"{k:13s} {fmt(first(h, LIN))} {fmt(first(h, 1e-4))}"
              f"   L(2000) = {h[-1]:.2e}")

    print("\n== Mini-batches of two, full-batch loss")
    out = minibatch_runs()
    print("        " + "".join(f"{k:>9s}" for k in out))
    for t in (10, 100, 500, 2000):
        print(f"t={t:5d} " + "".join(f"{h[t]:9.1e}" for h in out.values()))
    print("to L*   " + "".join(f"{fmt(first(h, LIN)):>9s}"
                               for h in out.values()))

    ra, rr, La, Lr = net_rates()
    print("\n== Effective rates, mini-batches of two, eta = 0.01")
    for t in (0, 9, 99, 1999):
        print(f"t={t + 1:5d}  AdaGrad {ra[t].min():.4f}-{ra[t].max():.4f}"
              f"  RMSProp {rr[t].min():.3f}-{rr[t].max():.3f}")
    print(f"L near 2000: AdaGrad {La:.2e}  RMSProp {Lr:.2e}")

    print("\n== Sensitivity, full batch, eta = 0.01: steps to L*,"
          " mean L over 1001-2000")
    for (om, z), (k, mL) in sensitivity().items():
        print(f"omega={om:<6} zeta={z:<6g} {fmt(k)}  {mL:.1e}")
