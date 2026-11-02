"""Chapter 22: light and searched optimisers. Adafactor (Shazeer and Stern
2018), Lion (Chen et al. 2023) and Sophia (Liu et al. 2024) on the
students and the 2-2-1 network."""
import numpy as np
from students import H
from net221 import init, loss, backward, Xs, ys
from ch10_perweight import batches, LIN, first
from ch14_adam import OM1, OM2, ZETA

np.set_printoptions(precision=4, suppress=True)
W10, w20 = init()
MASK = {1: np.array([[0, 1, 1], [0, 1, 1.0]]), 2: np.array([0, 1, 1.0])}


# --------------------------- the methods ---------------------------------
def rms(A):
    return np.sqrt(np.mean(A * A))


def factored(G2):
    """Row sums k, column sums n and the rank-one rebuild k n^T / 1^T k."""
    k, n = G2.sum(axis=1), G2.sum(axis=0)
    return k, n, np.outer(k, n) / k.sum()


def adamw_step(P, g, st, t, eta, rho, mask):
    m, r = st.get("m", 0 * P), st.get("r", 0 * P)
    m = OM1 * m + (1 - OM1) * g
    r = OM2 * r + (1 - OM2) * g * g
    st["m"], st["r"] = m, r
    step = (m / (1 - OM1 ** t)) / (np.sqrt(r / (1 - OM2 ** t)) + ZETA)
    return P - eta * rho * mask * P - eta * step


def adafactor_step(P, g, st, t, eta, rho, mask, z1=1e-30, z2=1e-3, c=1.0,
                   relative=True):
    """Algorithms 4 and 5 of the paper: no first moment, decay 1 - t^-0.8,
    update clipping at c, step eta * max(z2, RMS(P)) if relative."""
    om2 = 1 - t ** -0.8
    G2 = g * g + z1
    if P.ndim == 2:
        st["k"] = om2 * st.get("k", 0) + (1 - om2) * G2.sum(axis=1)
        st["n"] = om2 * st.get("n", 0) + (1 - om2) * G2.sum(axis=0)
        R = np.outer(st["k"], st["n"]) / st["k"].sum()
    else:
        st["r"] = om2 * st.get("r", 0) + (1 - om2) * G2
        R = st["r"]
    U = g / np.sqrt(R)
    U = U / max(1.0, rms(U) / c)
    a = eta * max(z2, rms(P)) if relative else eta
    return P - eta * rho * mask * P - a * U


def lion_step(P, g, st, t, eta, rho, mask, om1=0.9, om2=0.99):
    m = st.get("m", 0 * P)
    c = om1 * m + (1 - om1) * g
    st["m"] = om2 * m + (1 - om2) * g
    return P - eta * (np.sign(c) + rho * mask * P)


def clip(z, a=1.0):
    return np.clip(z, -a, a)


def sophia_step(P, g, st, t, eta, rho, mask, om1=0.96, om2=0.99, s=0.01,
                zeta=1e-12):
    """The Hessian estimate h_hat is passed in st['hhat'] on update steps."""
    m = om1 * st.get("m", 0 * P) + (1 - om1) * g
    st["m"] = m
    if "hhat" in st:
        st["h"] = om2 * st.get("h", 0 * P) + (1 - om2) * st.pop("hhat")
    ratio = m / np.maximum(s * st["h"], zeta)
    st["frac"] = st.get("frac", []) + [np.mean(np.abs(ratio) < 1)]
    P = P - eta * rho * mask * P
    return P - eta * clip(ratio)


def hvp(W1, w2, U1, u2, X, t, eps=1e-4):
    """Hessian-vector product by central differences of the gradient."""
    Ga, ga = backward(W1 + eps * U1, w2 + eps * u2, X, t)
    Gb, gb = backward(W1 - eps * U1, w2 - eps * u2, X, t)
    return (Ga - Gb) / (2 * eps), (ga - gb) / (2 * eps)


STEP = {"adamw": adamw_step, "adafactor": adafactor_step,
        "lion": lion_step, "sophia": sophia_step}


def run(kind, eta, rho=0.0, steps=2000, b=None, seed=9, f=10, **kw):
    W1, w2 = W10.copy(), w20.copy()
    s1, s2 = {}, {}
    gen = batches(b, seed) if b else None
    rng = np.random.default_rng(seed + 1)
    hist = [loss(W1, w2)]
    for t in range(1, steps + 1):
        idx = next(gen) if gen else slice(None)
        X, y = Xs[idx], ys[idx]
        G1, g2 = backward(W1, w2, X, y)
        if kind == "sophia" and t % f == 1 % f:
            U1, u2 = rng.standard_normal(W1.shape), rng.standard_normal(3)
            HU1, hu2 = hvp(W1, w2, U1, u2, X, y)
            s1["hhat"], s2["hhat"] = U1 * HU1, u2 * hu2
        W1 = STEP[kind](W1, G1, s1, t, eta, rho, MASK[1], **kw)
        w2 = STEP[kind](w2, g2, s2, t, eta, rho, MASK[2], **kw)
        hist.append(loss(W1, w2))
    run.frac = np.mean(s1.get("frac", [0]) + s2.get("frac", [0]))
    return np.array(hist)


def to_star(h):
    i = first(h, LIN)
    return "never" if i is None else i


# state vectors (numbers stored besides weights and gradient) on the net
MEMORY = {"AdamW": 2 * 9, "Adafactor": (2 + 3) + 3, "Lion": 9,
          "Sophia": 2 * 9}


# --------------------------- hand examples -------------------------------
def hand_adafactor():
    G, _ = backward(W10, w20)
    G2 = G * G
    k, n, R = factored(G2)
    print("G =\n", G.round(4))
    print("G*G (x 1e4) =\n", (1e4 * G2).round(2))
    print("k (x 1e4) =", (1e4 * k).round(2), " n (x 1e4) =", (1e4 * n).round(2),
          " 1^T k (x 1e4) =", round(1e4 * k.sum(), 2))
    print("R tilde (x 1e4) =\n", (1e4 * R).round(2))
    print("relative error of R tilde:\n", ((R - G2) / G2).round(3))
    print("Adam step G/sqrt(G*G) =\n", (G / np.sqrt(G2)).round(3))
    U = G / np.sqrt(R)
    print("Adafactor U = G/sqrt(R tilde) =\n", U.round(3), " RMS(U) =",
          round(rms(U), 4))


def hand_lion(eta=0.01, rho=0.1, js=((0, 0), (1, 1))):
    """Two full-batch Lion steps on all weights; prints the entries of W1
    picked by js (a bias, which is not decayed, and a decayed weight)."""
    W1, w2 = W10.copy(), w20.copy()
    s1, s2 = {}, {}
    for t in (1, 2):
        G1, g2 = backward(W1, w2)
        m = s1.get("m", 0 * W1)
        c = 0.9 * m + 0.1 * G1
        new = lion_step(W1, G1, s1, t, eta, rho, MASK[1])
        for (i, j) in js:
            print(f"t={t} w[{i},{j}]: w={W1[i, j]:.4f} g={G1[i, j]:.5f} "
                  f"m_prev={m[i, j]:.6f} c={c[i, j]:.6f} "
                  f"m_new={s1['m'][i, j]:.6f} w_new={new[i, j]:.6f}")
        w2 = lion_step(w2, g2, s2, t, eta, rho, MASK[2])
        W1 = new


SIGNS = np.array([[1, -1, 1], [1, 1, -1], [-1, 1, 1.0]])


def hand_hutchinson(U=SIGNS):
    print("H =\n", H.round(4), "\ndiag(H) =", np.diag(H).round(4))
    est = []
    for u in U:
        Hu = H @ u
        e = u * Hu
        est.append(e)
        print("u =", u, " Hu =", Hu.round(4), " u*Hu =", e.round(4))
    est = np.array(est)
    print("mean of the three:", est.mean(axis=0).round(4))
    off = (H ** 2).sum(axis=1) - np.diag(H) ** 2
    print("variance per probe (signs):", off.round(3),
          " sd:", np.sqrt(off).round(3))
    print("variance per probe (Gaussian):", (off + 2 * np.diag(H) ** 2).round(3),
          " sd:", np.sqrt(off + 2 * np.diag(H) ** 2).round(3))
    all4 = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1.0]])
    print("mean over the four sign patterns:",
          np.mean([u * (H @ u) for u in all4], axis=0).round(4))


def hutch_path(n=200, seed=22):
    """Running means of the estimate for signs and for Gaussian probes."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in ("signs", "gauss"):
        if name == "signs":
            U = rng.choice([-1.0, 1.0], size=(n, 3))
        else:
            U = rng.standard_normal((n, 3))
        E = U * (U @ H)
        out[name] = np.cumsum(E, axis=0) / np.arange(1, n + 1)[:, None]
    return out


# --------------------------- experiments ---------------------------------
GRID = {"adamw": (0.01, 0.03, 0.1, 0.3),
        "adafactor": (0.01, 0.03, 0.1, 0.3),
        "lion": (0.003, 0.01, 0.03, 0.1),
        "sophia": (0.003, 0.01, 0.03, 0.1)}
RHO = {"adamw": 0.01, "adafactor": 0.0, "lion": 0.1, "sophia": 0.01}


def grids():
    print("full batch, steps to L* within 2000 steps")
    for kind, etas in GRID.items():
        row = [to_star(run(kind, e, RHO[kind])) for e in etas]
        print(f"  {kind:9s} rho={RHO[kind]:<5} eta={etas}: {row}")


def sophia_scale():
    print("Sophia full batch, eta=0.03: s, steps to L*, share unclipped")
    for s in (0.001, 0.01, 0.1, 1.0):
        h = run("sophia", 0.03, RHO["sophia"], s=s)
        print(f"  s={s:<6} {to_star(h)!s:>6}  {run.frac:.3f}")


BEST = {"adamw": 0.1, "adafactor": 0.03, "lion": 0.03, "sophia": 0.03}


def table():
    print("method    b  to L*  L(500)     L(2000)    mean last 200")
    for b in (None, 2):
        for kind, eta in BEST.items():
            h = run(kind, eta, RHO[kind], b=b)
            print(f"  {kind:9s} {b or 8}  {to_star(h)!s:>5}  {h[500]:.3e}  "
                  f"{h[-1]:.3e}  {h[-200:].mean():.3e}")


if __name__ == "__main__":
    print("L* (standard units) =", round(LIN, 6), " L(w0) =",
          round(loss(W10, w20), 4))
    print("== Adafactor: factored second moment at the start ==")
    hand_adafactor()
    print("== Lion: two full-batch steps (eta=0.01, rho=0.1) ==")
    hand_lion()
    print("== Hutchinson on the linear H ==")
    hand_hutchinson()
    p = hutch_path()
    print("running means after 200 probes: signs", p["signs"][-1].round(3),
          " gauss", p["gauss"][-1].round(3))
    print("== memory (stored numbers besides weights and gradient) ==")
    for k_, v in MEMORY.items():
        print(f"  {k_:10s} {v}")
    print("== grids ==")
    grids()
    sophia_scale()
    print("== table ==")
    table()
