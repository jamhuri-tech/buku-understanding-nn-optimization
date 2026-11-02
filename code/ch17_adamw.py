"""Chapter 17: decoupled weight decay, AdamW and SGDW (Loshchilov and Hutter 2019)."""
import numpy as np
from students import X, y, m, H, w_star, loss, grad
from net221 import init, flat, loss_flat, grad_flat
from ch15_schedules import cosine

np.set_printoptions(precision=4, suppress=True)
OM1, OM2, ZETA = 0.9, 0.999, 1e-8
E3 = np.array([0.0, 1.0, 1.0])            # linear model: intercept not decayed
E9 = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1.0])   # net: biases not decayed


def adamw(g, L, w, eta, rho, mask, steps, decoupled=True, om1=OM1, om2=OM2,
          zeta=ZETA, log=None):
    """Adam+L2 (decoupled=False) or AdamW; eta is a number or a function of t."""
    w = w.copy()
    mt, r = np.zeros(len(w)), np.zeros(len(w))
    hist = [L(w)]
    for t in range(steps):
        et = eta(t) if callable(eta) else eta
        gt = g(w) + (0.0 if decoupled else rho * mask * w)
        mt = om1 * mt + (1 - om1) * gt
        r = om2 * r + (1 - om2) * gt * gt
        mh = mt / (1 - om1 ** (t + 1))
        den = np.sqrt(r / (1 - om2 ** (t + 1))) + zeta
        dec = et * rho * mask * w if decoupled else et * rho * mask * w / den
        if log is not None:
            log.append(dict(w=w.copy(), g=gt.copy(), mh=mh.copy(),
                            den=den.copy(), dec=dec.copy()))
        w = w - et * mh / den - (dec if decoupled else 0.0)
        hist.append(L(w))
    return w, hist


def gd_l2(g, L, w, eta, rho, mask, steps):
    w = w.copy()
    for _ in range(steps):
        w = w - eta * (g(w) + rho * mask * w)
    return w


# ---------------------------------------------------------------- linear model
def lin_pen(rho):
    """Exact minimiser of L + rho/2 |w bar|^2 (raw features)."""
    return np.linalg.solve(H + rho * np.diag(E3), X.T @ y / m)


def gd_hand(rho=0.01, eta=0.02, w0=np.array([10.0, 5.0, 5.0])):
    gL = grad(w0)
    a = w0 - eta * (gL + rho * E3 * w0)
    b = (1 - eta * rho * E3) * w0 - eta * gL
    print("GD+L2 step", a, "decay form", b, "factors", 1 - eta * rho * E3)


def lin_hand(rho=0.01, eta=0.1, w0=np.array([10.0, 5.0, 5.0])):
    """One step of Adam+L2 and AdamW from w0."""
    print("w0", w0, "grad L", grad(w0), "rho E w0", rho * E3 * w0)
    for dec in (False, True):
        lg = []
        w1, _ = adamw(grad, loss, w0, eta, rho, E3, 1, dec, log=lg)
        d = lg[0]
        print("AdamW" if dec else "Adam+L2", "g", d["g"], "den", d["den"])
        print("  move", w1 - w0, "w1", w1)
        if dec:
            print("  decay term", d["dec"], "rate", eta * rho * E3)
        else:
            print("  loss part", eta * grad(w0) / d["den"])
            print("  decay term", eta * rho * E3 * w0 / d["den"],
                  "rate", eta * rho * E3 / d["den"])


def pl(w, rho=0.01):
    return loss(w) + 0.5 * rho * np.sum((E3 * w) ** 2)


def lin_table(rho=0.01, eta=0.1, T=20000):
    wp = lin_pen(rho)
    print("w* (no penalty)", w_star, "L %.4f pen %.4f" % (loss(w_star), pl(w_star)))
    print("w_rho exact", wp, "L %.4f pen %.4f" % (loss(wp), pl(wp)))
    w0 = np.zeros(3)
    res = {"GD+L2": gd_l2(grad, loss, w0, 0.02, rho, E3, T),
           "Adam+L2": adamw(grad, loss, w0, cosine(eta, T), rho, E3, T,
                            False)[0],
           "AdamW": adamw(grad, loss, w0, cosine(eta, T), rho, E3, T)[0]}
    for k, w in res.items():
        print(f"{k:8s}", w, "L %.4f pen %.4f to w_rho %.1e to w* %.1e"
              % (loss(w), pl(w), np.linalg.norm(w - wp),
                 np.linalg.norm(w - w_star)))
    return res


# ---------------------------------------------------------------- 1-D model
def one_d(h=1.0, a=2.0, rho=0.1, eta=0.01, steps=20000, sig=0.0, seed=0):
    """L = h/2 (w-a)^2 with gradient noise sig; last-half averages of w
    and of the divisor, for Adam+L2 (False) and AdamW (True)."""
    rng = np.random.default_rng(seed)
    out = {}
    for dec in (False, True):
        it = iter(rng.standard_normal(steps) * sig)
        gfun = lambda w: h * (w - a) + next(it)
        lg = []
        adamw(gfun, lambda w: 0.0, np.array([0.0]), eta, rho,
              np.array([1.0]), steps, dec, log=lg)
        ws = np.array([d["w"][0] for d in lg[steps // 2:]])
        dn = np.array([d["den"][0] for d in lg[steps // 2:]])
        out[dec] = (ws.mean(), dn.mean())
        print("1-D sig %.1f %s w %.4f den %.4f predicted %.4f" % (
            sig, "AdamW  " if dec else "Adam+L2", ws.mean(), dn.mean(),
            h * a / (h + rho * dn.mean()) if dec else h * a / (h + rho)))
    return out


def cap_hand(rho=5.0, eta=0.03, w=0.1, n=3):
    """AdamW on one weight whose Adam direction is -1 (a push upward)."""
    for t in range(n):
        w = w - eta * (-1 + rho * w)
        print("cap step", t + 1, "w %.5f" % w)
    print("fixed point 1/rho =", 1 / rho)


# ---------------------------------------------------------------- network
RHO_L2 = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
RHO_W = (0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 10.0)


def net_run(rho, dec, eta=0.03, T=2000, log=None):
    p0 = flat(*init())
    return adamw(grad_flat, loss_flat, p0, cosine(eta, T), rho, E9, T, dec,
                 log=log)


def net_sweep():
    rows = []
    for dec, rhos in ((False, RHO_L2), (True, RHO_W)):
        for rho in rhos:
            p, _ = net_run(rho, dec)
            rows.append((rho, dec, loss_flat(p), np.linalg.norm(E9 * p),
                         np.abs(E9 * p).max(), p))
    return rows


def eff_decay(rho=0.01, eta=0.03, T=2000):
    """Per-weight decay rate eta rho/(sqrt(rhat)+zeta) of Adam+L2 on the
    net, constant eta; rows are steps."""
    lg = []
    p0 = flat(*init())
    adamw(grad_flat, loss_flat, p0, eta, rho, E9, T, False, log=lg)
    return np.array([eta * rho / d["den"] for d in lg])


# ---------------------------------------------------------------- cross-entropy
TL = (y >= 65).astype(float)


def ce(w):
    z = X @ w
    return np.mean(np.logaddexp(0, z) - TL * z)


def gce(w):
    return X.T @ (1 / (1 + np.exp(-(X @ w))) - TL) / m


def ce_runs(eta=0.1, T=20000,
            runs=((0.01, False), (0.01, True), (0.3, True), (1.0, True))):
    out = {}
    for rho, dec in runs:
        w, _ = adamw(gce, ce, np.zeros(3), cosine(eta, T), rho, E3, T, dec)
        out[(rho, dec)] = w
        print("CE", "AdamW  " if dec else "Adam+L2", "rho %.2f" % rho, w,
              "CE %.2e pen(0.01) %.4f" % (ce(w), ce(w) + 0.005 * np.sum(
                  (E3 * w) ** 2)), "max %.4f" % np.abs(E3 * w).max())
    return out


if __name__ == "__main__":
    gd_hand()
    lin_hand()
    lin_table()
    for s in (0.0, 1.0):
        one_d(sig=s)
    cap_hand()
    for r in net_sweep():
        print("net %s rho %5.3f L %.3e (%.4f pts^2) |wbar| %.4f max %.4f"
              % ("AdamW  " if r[1] else "Adam+L2", r[0], r[2],
                 r[2] * 123.4375, r[3], r[4]))
    p, _ = net_run(0.0, False)
    print("net Adam rho 0 L %.3e |wbar| %.4f max %.4f" % (
        loss_flat(p), np.linalg.norm(E9 * p), np.abs(E9 * p).max()))
    ed = eff_decay()
    for t in (0, 10, 100, 1000, 1999):
        print("eff decay t=%d" % t, ed[t][E9 > 0])
    ce_runs()
    w, _ = adamw(gce, ce, np.zeros(3), 0.1, 0.01, E3, 100000)
    print("CE AdamW fixed eta 0.1, 1e5 steps", w)
