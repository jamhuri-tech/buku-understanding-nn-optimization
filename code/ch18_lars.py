"""Chapter 18: layer-wise scaling, LARS (You et al. 2017), LAMB (2019)."""
import numpy as np
from net221 import init, flat, unflat, loss_flat, grad_flat, sample_grads
from ch10_perweight import batches, first, LIN
from ch14_adam import OM1, OM2, ZETA

np.set_printoptions(precision=4, suppress=True)
LAYERS = (slice(0, 6), slice(6, 9))      # W1 (rows), then w2


def norms(p):
    return np.array([np.linalg.norm(p[s]) for s in LAYERS])


def trust(p, g, psi, rho=0.0):
    """Trust ratios q = psi ||w|| / (||g|| + rho ||w||), one per layer."""
    nw, ng = norms(p), norms(g)
    den = ng + rho * nw
    ok = (nw > 0) & (den > 0)
    return np.where(ok, psi * nw / np.where(ok, den, 1.0), 1.0)


def poly(t, T):
    """Polynomial decay (1 - t/T)^2 of the LARS paper; 1 when T is None."""
    return 1.0 if T is None else (1 - t / T) ** 2


def sgdm(g, L, p, eta, gam, steps, T=None):
    """Heavy ball v = gam v - eta g; mult rescales the stored layers."""
    p, v = p.copy(), np.zeros(len(p))
    hist = [L(p)]
    for t in range(steps):
        v = gam * v - eta * poly(t, T) * g(p)
        p = p + v
        hist.append(L(p))
        if not np.isfinite(hist[-1]) or hist[-1] > 1e6:
            break
    return p, hist


def lars(g, L, p, eta, gam, steps, psi=1.0, rho=0.0, T=None, log=None):
    """LARS: v = gam v - eta q (g + rho w) layer by layer, then w = w + v."""
    p, v = p.copy(), np.zeros(len(p))
    hist = [L(p)]
    for t in range(steps):
        gt = g(p)
        q = trust(p, gt, psi, rho)
        for s, ql in zip(LAYERS, q):
            v[s] = gam * v[s] - eta * poly(t, T) * ql * (gt[s] + rho * p[s])
        if log is not None:
            log.append((p.copy(), gt.copy(), q.copy(), v.copy()))
        p = p + v
        hist.append(L(p))
    return p, hist


def adam_l(g, L, p, eta, steps, rho=0.0, layer=False, om1=OM1, om2=OM2,
           zeta=ZETA, T=None, log=None):
    """AdamW direction u = mhat/(sqrt rhat + zeta) + rho w.

    layer=False: AdamW step w - eta u.  layer=True: LAMB with phi(z) = z,
    w - eta ||w|| / ||u|| u for each layer.
    """
    p = p.copy()
    mt, r = np.zeros(len(p)), np.zeros(len(p))
    hist = [L(p)]
    for t in range(steps):
        gt = g(p)
        mt = om1 * mt + (1 - om1) * gt
        r = om2 * r + (1 - om2) * gt * gt
        u = (mt / (1 - om1 ** (t + 1))) / (np.sqrt(r / (1 - om2 ** (t + 1)))
                                          + zeta) + rho * p
        s = -eta * poly(t, T) * u
        if layer:
            for sl in LAYERS:
                s[sl] *= np.linalg.norm(p[sl]) / np.linalg.norm(u[sl])
        if log is not None:
            log.append((p.copy(), gt.copy(), u.copy(), s.copy()))
        p = p + s
        hist.append(L(p))
        if not np.isfinite(hist[-1]) or hist[-1] > 1e6:
            break
    return p, hist


def hand():
    """Norms, gradients, trust ratios and one LARS step at the start."""
    W1, w2 = init()
    p = flat(W1, w2)
    g = grad_flat(p)
    G1, g2 = unflat(g)
    print("W1 =\n", W1, "\nw2 =", w2)
    print("G1 =\n", G1.round(4), "\ng2 =", g2.round(4))
    nw, ng = norms(p), norms(g)
    print("||W1||_F^2 =", round((W1 ** 2).sum(), 4), " ||w2||^2 =",
          round((w2 ** 2).sum(), 4))
    print("||G1||_F^2 =", round((G1 ** 2).sum(), 6), " ||g2||^2 =",
          round((g2 ** 2).sum(), 6))
    print("||w|| =", nw.round(4), " ||g|| =", ng.round(4),
          " ratio ||w||/||g|| =", (nw / ng).round(3))
    for psi in (1.0,):
        q = trust(p, g, psi)
        print(f"psi = {psi}: q = {q.round(4)}")
    eta = 0.05
    print("q * g, layer 1 =\n", (q[0] * G1).round(4), "\nq * g, layer 2 =",
          (q[1] * g2).round(4))
    q = trust(p, g, 1.0)
    p1 = p.copy()
    for s, ql in zip(LAYERS, q):
        p1[s] = p[s] - eta * ql * g[s]
    W1n, w2n = unflat(p1)
    print("LARS step eta = 0.05, psi = 1, first step (v = 0):")
    print("W1 ->\n", W1n.round(4), "\nw2 ->", w2n.round(4))
    d = p1 - p
    print("relative change per layer =", (norms(d) / nw).round(4))
    print("L(w0) =", round(loss_flat(p), 4), " L(w1) LARS =",
          round(loss_flat(p1), 4), " L(w1) GD =",
          round(loss_flat(p - eta * g), 4))
    print("GD relative change per layer =",
          (eta * ng / nw).round(4))
    lamb = p.copy()
    u = np.sign(g)
    for s in LAYERS:
        lamb[s] = p[s] - eta * norms(p)[LAYERS.index(s)] * u[s] / \
            np.linalg.norm(u[s])
    print("LAMB first step (u = sign g up to zeta), eta = 0.05:")
    print("move per entry =", (eta * nw / np.sqrt([6, 3])).round(5))
    W1n, w2n = unflat(lamb)
    print("W1 ->\n", W1n.round(4), "\nw2 ->", w2n.round(4))
    chk = adam_l(grad_flat, loss_flat, p, eta, 1, layer=True)[0]
    print("check with adam_l:", np.abs(chk - lamb).max() < 1e-6,
          " L(w1) LAMB =", round(loss_flat(lamb), 4))
    return p, g, q, p1


def noise_scale(p):
    """Simple noise scale tr(Sigma)/||g||^2 from the per-student gradients."""
    G = sample_grads(p)
    g = G.mean(axis=0)
    return ((G - g) ** 2).sum(axis=1).mean() / (g @ g)


def ratio_path(eta=0.1, steps=3000):
    """||w||/||g|| for each layer along full-batch GD."""
    p = flat(*init())
    out = []
    for t in range(steps + 1):
        g = grad_flat(p)
        out.append(norms(p) / norms(g))
        p = p - eta * g
    return np.array(out)


GRID = {"SGDM": (0.03, 0.1, 0.3, 1.0), "LARS": (0.01, 0.02, 0.05, 0.1),
        "AdamW": (0.01, 0.03, 0.1, 0.3), "LAMB": (0.01, 0.03, 0.1, 0.3)}


def run(name, g, L, p0, eta, steps, gam=0.9):
    """One method with the polynomial decay over all its steps."""
    if name == "SGDM":
        return sgdm(g, L, p0, eta, gam, steps, T=steps)[1]
    if name == "LARS":
        return lars(g, L, p0, eta, gam, steps, T=steps)[1]
    return adam_l(g, L, p0, eta, steps, layer=(name == "LAMB"), T=steps)[1]


def tune(epochs=400):
    """Full batch (k = 8): the rate of each grid with fewest steps to L*."""
    p0 = flat(*init())
    best = {}
    for name, grid in GRID.items():
        res = []
        for eta in grid:
            h = run(name, grad_flat, loss_flat, p0, eta, epochs)
            hit = first(h, LIN)
            res.append((hit if hit is not None else 10 ** 9, eta, h[-1]))
            print(f"  {name:6s} eta = {eta:<5g} steps to L* = "
                  f"{str(hit or '--'):>4s}  final L = {h[-1]:.2e}")
        best[name] = min(res)[1]
    print("  chosen:", best)
    return best


def batch_table(base, epochs=400):
    """k = 1, 2, 4, 8 with the rates scaled linearly from k = 8."""
    p0 = flat(*init())
    rows = []
    for k in (1, 2, 4, 8):
        steps = epochs * 8 // k
        res = {}
        for name, e8 in base.items():
            gen = batches(k)
            gb = (lambda gen: lambda p: grad_flat(p, next(gen)))(gen)
            h = run(name, gb, loss_flat, p0, e8 * k / 8, steps)
            hit = first(h, LIN)
            res[name] = (e8 * k / 8, None if hit is None else hit * k / 8,
                         h[-1], h)
        rows.append((k, steps, res))
    return rows


def show_batch(rows):
    print(f"linear L* (standard units) = {LIN:.6f}")
    for k, steps, res in rows:
        print(f"k = {k}, steps = {steps}")
        for name, (eta, ep, fin, _) in res.items():
            e = "--" if ep is None else f"{ep:g}"
            print(f"  {name:6s} eta = {eta:<8g} epochs to L* = {e:>7s}"
                  f"  final L = {fin:.2e}")


def rescaled(base, c=10.0, steps=400):
    """Stored W1 = c W1 used as W/c; stored w2 = w2/c used as c w.

    The net computes the same function; the gradients of the stored
    weights are G1/c and c g2.
    """
    p0 = flat(*init())
    M = np.r_[np.full(6, c), np.full(3, 1 / c)]      # stored = M * true

    def L(s):
        return loss_flat(s / M)

    def g(s):
        return grad_flat(s / M) / M

    out = {}
    for name, eta in base.items():
        out[name] = (eta, run(name, grad_flat, loss_flat, p0, eta, steps))
        out[name + " resc."] = (eta, run(name, g, L, M * p0, eta, steps))
    for eta in (0.03, 0.01, 0.001):
        out[f"SGDM resc. {eta}"] = (eta, run("SGDM", g, L, M * p0, eta,
                                             steps))
    return out


def show_rescaled(out):
    print("rescaled layers (c = 10), full batch, 400 steps")
    for k, (eta, h) in out.items():
        hit = first(h, LIN)
        print(f"  {k:17s} eta = {eta:<6g} steps to L* = "
              f"{str(hit or '--'):>4s}  L(end) = {h[-1]:.2e}"
              f"  ({len(h) - 1} steps)")


if __name__ == "__main__":
    print("== (a) Hand example at the start")
    hand()
    print("\n== Noise scale tr(Sigma)/||g||^2 along GD, eta = 0.1")
    p = flat(*init())
    for t in range(3001):
        if t in (0, 10, 100, 1000, 3000):
            print(f"t = {t:5d}: B = {noise_scale(p):10.3f}"
                  f"  L = {loss_flat(p):.2e}")
        p = p - 0.1 * grad_flat(p)
    print("\n== (b) ||w||/||g|| along GD, eta = 0.1")
    R = ratio_path()
    for t in (0, 10, 20, 50, 100, 500, 1000, 2000, 3000):
        print(f"t = {t:5d}: ratio W1 = {R[t, 0]:9.3f}  w2 = {R[t, 1]:9.3f}"
              f"  W1/w2 = {R[t, 0] / R[t, 1]:7.3f}")
    print("\n== (c) Rates at k = 8, polynomial decay, 400 steps")
    base = tune()
    print("\n== (c) Batch sizes, 400 epochs, linear scaling from k = 8")
    show_batch(batch_table(base))
    print("\n== (d) Rescaled layers")
    show_rescaled(rescaled(base))
