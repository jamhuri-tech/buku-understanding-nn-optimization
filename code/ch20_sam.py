"""Chapter 20: sharpness-aware minimization, SAM (Foret et al. 2021),
and the adaptive variant ASAM (Kwon et al. 2021)."""
import numpy as np
from net221 import init, flat, loss_flat, grad_flat, hess_flat
from ch10_perweight import batches, first, LIN
from ch14_adam import OM1, OM2, ZETA

np.set_printoptions(precision=4, suppress=True)
P0 = flat(*init())


def ascent(g, r, p=None, eta_a=0.0):
    """First-order inner solution r g/||g||; ASAM scales by T = |p| + eta_a."""
    if p is None:
        return r * g / np.linalg.norm(g)
    T = np.abs(p) + eta_a
    return r * T * T * g / np.linalg.norm(T * g)


def optimise(eta, steps, r=0.0, kind="sgd", b=None, seed=9, p=P0,
             asam=None):
    """GD/SGD or Adam, with the SAM gradient when r > 0.

    The ascent and the descent use the same mini-batch. Returns the last
    weights, the losses L(w_t) and the number of gradient evaluations."""
    p = p.copy()
    gen = batches(b, seed) if b else None
    mt, v = np.zeros(len(p)), np.zeros(len(p))
    hist, evals = [loss_flat(p)], 0
    for t in range(1, steps + 1):
        idx = next(gen) if gen else None
        g = grad_flat(p, idx)
        evals += 1
        if r > 0:
            e = ascent(g, r, p if asam is not None else None, asam or 0.0)
            g = grad_flat(p + e, idx)
            evals += 1
        if kind == "adam":
            mt = OM1 * mt + (1 - OM1) * g
            v = OM2 * v + (1 - OM2) * g * g
            g = (mt / (1 - OM1 ** t)) / (np.sqrt(v / (1 - OM2 ** t)) + ZETA)
        p = p - eta * g
        hist.append(loss_flat(p))
    return p, np.array(hist), evals


def sharp(p):
    """Top eigenvalue and trace of the Hessian."""
    ev = np.linalg.eigvalsh(hess_flat(p))
    return ev[-1], ev.sum()


def robust(p, s=0.1, n=2000, seed=9):
    """Mean rise of the loss under random N(0, s^2 I) weight changes."""
    rng = np.random.default_rng(seed)
    L0 = loss_flat(p)
    return np.mean([loss_flat(p + s * rng.standard_normal(len(p))) - L0
                    for _ in range(n)])


# ------------------------- hand examples ---------------------------------
def lin_step(r=0.1, eta=0.1):
    """One SAM step on the standardised linear model from w = 0."""
    from ch19_stability import lin_parts
    Hs, bs = lin_parts()
    w = np.zeros(3)
    g = Hs @ w - bs
    e = ascent(g, r)
    ge = Hs @ (w + e) - bs
    return Hs, bs, g, e, ge, w - eta * ge, w - eta * g


def bounce(h=2.0, eta=0.1, r=0.05, w=0.1, steps=8):
    """SAM on L = h w^2/2: w <- (1 - eta h) w - eta h r sign(w)."""
    out = [w]
    for _ in range(steps):
        w = (1 - eta * h) * w - eta * h * r * np.sign(w)
        out.append(w)
    return np.array(out), eta * h * r / (2 - eta * h)


# --------------------- a valley of minima (figure) ------------------------
def a_val(u):
    """Curvature across the valley; sharp for u < 0, flat for u > 0."""
    return 1 + 8 / (1 + np.exp(2 * u))


def da_val(u):
    return -16 * np.exp(2 * u) / (1 + np.exp(2 * u)) ** 2


def L_val(w):
    return 0.5 * a_val(w[0]) * w[1] ** 2


def g_val(w):
    return np.array([0.5 * da_val(w[0]) * w[1] ** 2, a_val(w[0]) * w[1]])


def valley(w0, r, eta=0.1, steps=1000):
    w = np.array(w0, float)
    path = [w.copy()]
    for _ in range(steps):
        g = g_val(w)
        if r > 0:
            g = g_val(w + ascent(g, r))
        w = w - eta * g
        path.append(w.copy())
    return np.array(path)


RUNS = (("GD", 0.1, 0.0, "sgd"), ("SAM", 0.1, 0.1, "sgd"),
        ("Adam", 0.01, 0.0, "adam"), ("Adam-SAM", 0.01, 0.1, "adam"))
EVALS = 4000


def net_run(eta, r, kind, b=None):
    """Same budget of gradient evaluations for every method."""
    steps = EVALS // 2 if r > 0 else EVALS
    return optimise(eta, steps, r, kind, b)


if __name__ == "__main__":
    print("== Hand example (a): one SAM step, standardised linear model, w=0")
    Hs, bs, g, e, ge, w1, wgd = lin_step()
    print("H =\n", Hs, "\nb =", bs)
    print("g =", g, " ||g|| =", f"{np.linalg.norm(g):.4f}")
    print("eps =", e, " ||eps|| =", f"{np.linalg.norm(e):.4f}")
    print("g at w+eps =", ge)
    print("SAM w1 =", w1, " GD w1 =", wgd)
    from ch19_stability import lin_parts
    Ll = lambda w: 0.5 * w @ Hs @ w - bs @ w + 0.5
    print(f"L(0) = {Ll(np.zeros(3)):.4f}  L(eps) = {Ll(e):.4f}"
          f"  L(SAM w1) = {Ll(w1):.4f}  L(GD w1) = {Ll(wgd):.4f}")
    lam = np.linalg.eigvalsh(Hs)
    print("eigenvalues of H:", lam)
    ws = np.linalg.solve(Hs, bs)
    q = np.linalg.eigh(Hs)[1][:, -1]
    print("w* =", ws, " q_max =", q)
    for r in (0.1, 0.5):
        print(f"r={r}: r^2 lam_max/2 = {0.5 * r * r * lam[-1]:.5f}"
              f"  L(w*+r q)-L(w*) = {Ll(ws + r * q) - Ll(ws):.5f}")

    print("\n== Hand example (b): SAM on L = h w^2/2, h=2, eta=0.1, r=0.05")
    ws_, amp = bounce()
    print("w_t:", ws_.round(5), " amplitude =", f"{amp:.5f}")

    print("\n== Valley L = a(w1) w2^2 / 2, eta = 0.1, 1000 steps")
    for w0 in ((-2.0, 1.5), (-1.0, -1.2)):
        for r in (0.0, 0.2):
            P = valley(w0, r)
            print(f"start {w0} r={r}: end {P[-1].round(3)}"
                  f"  a = {a_val(P[-1][0]):.3f}  L = {L_val(P[-1]):.2e}")
    print("bounce prediction at a=1.173:",
          f"{0.1 * 1.173 * 0.2 / (2 - 0.1 * 1.173):.4f}")

    print("\n== Net (c): same budget of", EVALS, "gradient evaluations")
    print("method     k  evals-to-L*  L(end)    lam9    trH    rise(0.1)")
    for b in (None, 2):
        for name, eta, r, kind in RUNS:
            p, h, ev = net_run(eta, r, kind, b)
            lm, tr = sharp(p)
            hit = first(h, LIN)
            hit = None if hit is None else hit * (2 if r > 0 else 1)
            print(f"{name:9s} {b or 8:2d}  {hit!s:>5}  {h[-1]:.2e}"
                  f"  {lm:.4f}  {tr:.4f}  {robust(p):.4f}"
                  f"  pred {0.5 * 0.01 * tr:.4f}")

    print("\n== Radius (d): SAM, full batch, eta = 0.1, 2000 steps")
    for r in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3):
        p, h, ev = optimise(0.1, 2000, r)
        lm, tr = sharp(p)
        print(f"r={r:<5} L={h[-1]:.2e}  lam9={lm:.4f}  trH={tr:.4f}"
              f"  rise={robust(p):.4f}  hit={first(h, LIN)}"
              f"  max|w|={np.abs(p).max():.3f}")
