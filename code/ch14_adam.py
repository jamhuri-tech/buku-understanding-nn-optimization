"""Chapter 14: Adam and AdaMax (Kingma and Ba 2014), Nadam (Dozat 2016)."""
import numpy as np
from students import H, w_star, loss, grad
from net221 import init, flat, loss_flat, grad_flat, train
from ch10_perweight import batches, first, LIN
from ch12_adagrad import adagrad, gd, heavy_ball, fmt
from ch13_rmsprop import rmsprop

np.set_printoptions(precision=4, suppress=True)
LS = loss(w_star)                 # linear L*, points^2
OM1, OM2, ZETA = 0.9, 0.999, 1e-8  # defaults of Kingma and Ba (2014)


def adam(g, L, w, eta, om1=OM1, om2=OM2, steps=1, zeta=ZETA, target=None,
         log=None, kind="adam", correct=True):
    """Adam, AdaMax (kind='adamax') or Nadam (kind='nadam'); t starts at 0.

    correct=False drops both bias corrections (Adam only).
    """
    w = w.copy()
    mt, r = np.zeros(len(w)), np.zeros(len(w))
    hist, hit = [L(w)], None
    for t in range(steps):
        gt = g(w)
        mt = om1 * mt + (1 - om1) * gt
        c1 = 1 - om1 ** (t + 1) if correct else 1.0
        if kind == "adamax":
            r = np.maximum(om2 * r, np.abs(gt))
            mh, den = mt / c1, r + zeta
        else:
            r = om2 * r + (1 - om2) * gt * gt
            c2 = 1 - om2 ** (t + 1) if correct else 1.0
            if kind == "nadam":
                mh = om1 * mt / (1 - om1 ** (t + 2)) + (1 - om1) * gt / c1
            else:
                mh = mt / c1
            den = np.sqrt(r / c2) + zeta
        s = -eta * mh / den
        if log is not None:
            log.append((w.copy(), gt.copy(), mt.copy(), r.copy(), mh.copy(),
                        den.copy(), s.copy()))
        w = w + s
        hist.append(L(w))
        if hit is None and target is not None and hist[-1] <= target:
            hit = t + 1
    return w, hit, hist


def nag(g, L, w, eta, gam, steps):
    """Nesterov, v = gam v - eta g(w + gam v), as in Chapter 8."""
    w, v = w.copy(), np.zeros(len(w))
    hist = [L(w)]
    for _ in range(steps):
        v = gam * v - eta * g(w + gam * v)
        w = w + v
        hist.append(L(w))
    return w, None, hist


def const_table(om1=OM1, om2=OM2, ts=(0, 1, 2, 4, 9, 19, 99, 999, 2999)):
    """Constant gradient g: move/eta without correction (with it, 1)."""
    return [(t, 1 - om1 ** (t + 1), 1 - om2 ** (t + 1),
             (1 - om1 ** (t + 1)) / np.sqrt(1 - om2 ** (t + 1))) for t in ts]


def linear_runs(steps=20000):
    w0, T = np.zeros(3), LS + 0.01
    out = {"GD 0.03": gd(grad, loss, w0, 0.03, steps, T),
           "AdaGrad 10": adagrad(grad, loss, w0, 10, steps, target=T)[:3],
           "RMSProp 0.01": rmsprop(grad, loss, w0, 0.01, steps=steps,
                                   target=T)[:3]}
    for kind in ("adam", "adamax", "nadam"):
        for eta in (0.1, 1.0):
            out[f"{kind} {eta}"] = adam(grad, loss, w0, eta, steps=steps,
                                        target=T, kind=kind)
    return out


def net_runs(steps=2000):
    p0 = flat(*init())
    out = {"GD": train(0.1, steps)[2],
           "HB": train(0.1, steps, 0.9)[2],
           "NAG": train(0.1, steps, 0.9, nesterov=True)[2],
           "RMSProp": rmsprop(grad_flat, loss_flat, p0, 0.01,
                              steps=steps)[2],
           "Adam 0.01": adam(grad_flat, loss_flat, p0, 0.01, steps=steps)[2]}
    for kind in ("adam", "adamax", "nadam"):
        out[f"{kind} 0.03"] = adam(grad_flat, loss_flat, p0, 0.03,
                                   steps=steps, kind=kind)[2]
    return out


def minibatch_runs(b=2, steps=2000, eta=0.03):
    """Seven methods on the same mini-batches of b students."""
    p0 = flat(*init())
    runs = {"SGD": lambda g: gd(g, loss_flat, p0, 0.1, steps),
            "HB": lambda g: heavy_ball(g, loss_flat, p0, 0.1, 0.9, steps),
            "NAG": lambda g: nag(g, loss_flat, p0, 0.1, 0.9, steps),
            "RMSProp": lambda g: rmsprop(g, loss_flat, p0, 0.01,
                                         steps=steps)}
    for kind in ("adam", "adamax", "nadam"):
        runs[kind] = (lambda k: lambda g: adam(g, loss_flat, p0, eta,
                                               steps=steps, kind=k))(kind)
    out = {}
    for k, run in runs.items():
        gen = batches(b)
        out[k] = run(lambda p: grad_flat(p, next(gen)))[2]
    return out


def bias_runs(eta=0.03, steps=300):
    """Adam on the net with and without bias correction: losses, moves."""
    p0 = flat(*init())
    out = {}
    for c in (True, False):
        log = []
        h = adam(grad_flat, loss_flat, p0, eta, steps=steps, log=log,
                 correct=c)[2]
        out[c] = (np.array(h), np.array([np.abs(x[6]).max() for x in log]))
    return out


def sensitivity(steps=2000):
    """Full-batch net, Adam: steps to linear L*, mean loss over 1001-2000."""
    p0 = flat(*init())
    out = {}
    for eta in (0.001, 0.01, 0.03, 0.1, 0.3):
        for z in (1e-8, 1e-4, 1e-2, 1e-1):
            h = adam(grad_flat, loss_flat, p0, eta, steps=steps, zeta=z)[2]
            out[(eta, z)] = (first(h, LIN), np.mean(h[1001:]))
    return out


if __name__ == "__main__":
    print("== Constant gradient, omega1 = 0.9, omega2 = 0.999:"
          " 1-om1^(t+1), 1-om2^(t+1), uncorrected move/eta")
    for t, a, b, s in const_table():
        print(f"t={t:5d}  {a:.4f}  {b:.6f}  {s:.4f}")

    print("\n== Hand examples: AdaMax on g = 3, -1, 0.5; Nadam factors")
    m, u = 0.0, 0.0
    for t, gt in enumerate((3.0, -1.0, 0.5)):
        m = OM1 * m + (1 - OM1) * gt
        u = max(OM2 * u, abs(gt))
        mh = m / (1 - OM1 ** (t + 1))
        print(f"t={t}  m = {m:.4f}  u = {u:.6f}  mhat = {mh:.4f}"
              f"  move/eta = {-mh / u:.4f}")
    for t in range(3):
        f = (OM1 * (1 - OM1 ** (t + 1)) / (1 - OM1 ** (t + 2))
             + (1 - OM1) / (1 - OM1 ** (t + 1)))
        print(f"Nadam, constant g, t={t}: mbar/g = {f:.4f}")

    print("\n== Adam, first two steps on the raw linear model,"
          " eta = 0.01, omega1 = 0.9, omega2 = 0.999")
    log = []
    adam(grad, loss, np.zeros(3), 0.01, steps=2, log=log)
    for t, (w, g, mt, r, mh, den, s) in enumerate(log):
        print(f"t={t} w = {w}  L = {loss(w):.4f}")
        print(f"   g = {g}")
        print(f"   m = {mt}")
        print(f"   r = {r}")
        print(f"   mhat = {mh}")
        print(f"   rhat = {(den - ZETA) ** 2}")
        print(f"   move = {np.array2string(s, precision=7)}")
    w2 = log[1][0] + log[1][6]
    print(f"t=2 w = {w2}  L = {loss(w2):.4f}")
    for kind in ("adamax", "nadam"):
        log = []
        adam(grad, loss, np.zeros(3), 0.01, steps=2, log=log, kind=kind)
        print(f"{kind}: moves {log[0][6]}  {log[1][6]}")

    print("\n== Linear model, raw features, steps to L* + 0.01")
    print(f"L* = {LS:.4f}")
    for k, r in linear_runs().items():
        h = r[2]
        print(f"{k:13s} {fmt(r[1])}   gap(20000) = {h[-1] - LS:.3e}"
              f"   max gap 10001-20000 = {max(h[10001:]) - LS:.3e}")

    print("\n== Adam, eta = 0.1: divisor sqrt(rhat) + zeta and the"
          " largest gap in the next 500 steps")
    log = []
    h = np.array(adam(grad, loss, np.zeros(3), 0.1, steps=20000,
                      log=log)[2]) - LS
    for t in (9000, 10000, 11000, 11500, 12000, 13000, 15000, 17000):
        print(f"t={t:5d}  divisor = {np.array2string(log[t][5], precision=4)}"
              f"  max gap = {max(h[t:t + 500]):.1e}")

    print("\n== 2-2-1 network, full batch")
    for k, h in net_runs().items():
        print(f"{k:13s} {fmt(first(h, LIN))} {fmt(first(h, 1e-4))}"
              f"   L(2000) = {h[-1]:.2e}")

    print("\n== Mini-batches of two, full-batch loss")
    out = minibatch_runs()
    names = {"adam": "Adam", "adamax": "AdaMax", "nadam": "Nadam"}
    print("       " + "".join(f"{names.get(k, k):>8s}" for k in out))
    for t in (10, 100, 500, 2000):
        print(f"t={t:4d} " + "".join(f"{h[t]:8.1e}" for h in out.values()))
    print("to L*  " + "".join(f"{fmt(first(h, LIN)):>8s}"
                              for h in out.values()))

    print("\n== Bias correction on the net, eta = 0.03")
    br = bias_runs()
    for c, (h, mv) in br.items():
        print(f"corrected={c!s:5s} largest move t=0,1,9: "
              f"{mv[0]:.4f} {mv[1]:.4f} {mv[9]:.4f}  max {mv.max():.4f}"
              f" at t={mv.argmax()}  L(50) = {h[50]:.3e}"
              f"  L(300) = {h[300]:.3e}")
    p0 = flat(*init())
    for eta in (0.1, 0.3):
        for c in (True, False):
            h = adam(grad_flat, loss_flat, p0, eta, steps=2000,
                     correct=c)[2]
            print(f"eta={eta} corrected={c!s:5s} to L* {fmt(first(h, LIN))}"
                  f"  L(2000) = {h[-1]:.2e}")

    print("\n== Sensitivity, full batch: steps to L*, mean L over 1001-2000")
    for (eta, z), (k, mL) in sensitivity().items():
        print(f"eta={eta:<6g} zeta={z:<6g} {fmt(k)}  {mL:.1e}")
