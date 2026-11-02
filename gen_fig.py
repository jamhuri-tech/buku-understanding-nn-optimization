"""Generate every Matplotlib figure of the book into fig/ as PDF.

One function per figure, named chNN_name(), which calls
save(fig, "chNN-name"). Functions named chNN_* run automatically;
`python3 gen_fig.py ch01` runs only the matching ones.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).parent / "fig"
sys.path.insert(0, str(Path(__file__).parent / "code"))

# Colours follow preamble.tex.
BLUE = "#1B3B6F"
GREEN = "#1E6F5C"
ORANGE = "#B85C00"
RED = "#9B1B30"
GREY = "#5A6472"
GREY_RULE = "#C9CED6"



plt.rcParams.update({
    "font.size": 8.5,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.edgecolor": GREY_RULE,
    "axes.labelcolor": GREY,
    "axes.titlesize": 9,
    "axes.titlecolor": BLUE,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "text.color": GREY,
    "grid.color": GREY_RULE,
    "legend.frameon": False,
    "figure.dpi": 300,
})




def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def ch00_taylor():
    """f(w) = ln(1 + e^w) - 0.6 w with its first- and second-order models at w = 1."""
    f = lambda w: np.log1p(np.exp(w)) - 0.6 * w
    s = lambda w: 1 / (1 + np.exp(-w))
    a = 1.0
    d1 = s(a) - 0.6
    d2 = s(a) * (1 - s(a))
    w = np.linspace(-3, 4, 300)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(w, f(w), color=BLUE, lw=1.4, label=r"$f(w)$")
    ax.plot(w, f(a) + d1 * (w - a), color=ORANGE, lw=1, ls="--",
            label="first-order model")
    ax.plot(w, f(a) + d1 * (w - a) + 0.5 * d2 * (w - a) ** 2, color=GREEN,
            lw=1, label="second-order model")
    ax.plot([a], [f(a)], "o", color=RED, ms=3.5)
    ax.annotate(r"$w = a$", (a, f(a)), xytext=(a + 0.25, f(a) - 0.45),
                color=RED)
    ax.set_xlabel(r"$w$")
    ax.set_ylim(-0.5, 2.5)
    ax.grid(lw=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              handlelength=1.5, columnspacing=1.0)
    save(fig, "ch00-taylor")


def ch00_contour():
    """Contours of f(w) = w^T H w / 2 with H = [[3,1],[1,3]], eigenvectors and a gradient."""
    H = np.array([[3.0, 1.0], [1.0, 3.0]])
    g1, g2 = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-2, 2, 300))
    F = 0.5 * (H[0, 0] * g1**2 + 2 * H[0, 1] * g1 * g2 + H[1, 1] * g2**2)
    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    ax.contour(g1, g2, F, levels=[0.25, 1, 2.25, 4, 6.25], colors=GREY,
               linewidths=0.6)
    for v, lab in (([1, 1], r"$\mathbf{q}_2,\ \lambda_2 = 4$"),
                   ([1, -1], r"$\mathbf{q}_1,\ \lambda_1 = 2$")):
        v = np.array(v) / np.sqrt(2)
        ax.annotate("", xy=1.6 * v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))
        ax.text(2.08, 1.6 * v[1], lab, color=GREEN, va="center",
                clip_on=False)
    p = np.array([1.0, 0.0])
    gr = H @ p
    ax.plot(*p, "o", color=RED, ms=3)
    ax.annotate("", xy=p - 0.25 * gr, xytext=p,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    ax.text(0.45, 0.13, r"$-\nabla f$", color=RED)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    save(fig, "ch00-contour")


def ch01_loss():
    """Loss of gradient descent on the students for two step sizes."""
    from students import loss, gd, w_star
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    t = np.arange(10001)
    for eta, col in ((0.03, BLUE), (0.02, ORANGE)):
        L = np.array([loss(w) for w in gd(eta, 10000)])
        ax.loglog(t[1:], L[1:] - loss(w_star), color=col, lw=1.2,
                  label=rf"$\eta = {eta}$")
    ax.axvline(10, color=GREY, lw=0.7, ls="--")
    ax.text(12, 150, "10 steps", color=GREY)
    ax.set_xlabel("step $t$")
    ax.set_ylabel(r"$L(\mathbf{w}_t) - L(\mathbf{w}^\star)$")
    ax.grid(lw=0.3, which="major")
    ax.legend(loc="lower left")
    save(fig, "ch01-loss")


def ch01_modes():
    """Distance to the minimiser along each eigenvector of H."""
    from students import gd, H, w_star
    lam, Q = np.linalg.eigh(H)
    P = np.array(gd(0.03, 10000)) - w_star
    C = np.abs(P @ Q)
    t = np.arange(1, 10001)
    fig, ax = plt.subplots(figsize=(2.9, 2.3))
    cols = (RED, GREEN, BLUE)
    labs = (r"flat, $\lambda_1 = 0.024$", r"middle, $\lambda_2 = 1.20$",
            r"steep, $\lambda_3 = 58.0$")
    for j in range(3):
        ax.loglog(t, np.maximum(C[1:, j], 1e-12), color=cols[j], lw=1.2,
                  label=labs[j])
    ax.set_ylim(1e-8, 1e2)
    ax.set_xlabel("step $t$")
    ax.set_ylabel("distance along $\\mathbf{q}_j$")
    ax.grid(lw=0.3, which="major")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(fig, "ch01-modes")


def ch02_shapes():
    """Bowl, trough and saddle: contours of three quadratic forms."""
    g1, g2 = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-2, 2, 300))
    cases = ((r"bowl, $\lambda = 1, 3$", 0.5 * (g1**2 + 3 * g2**2)),
             (r"trough, $\lambda = 0, 2$", 0.5 * (g1 + g2) ** 2),
             (r"saddle, $\lambda = -1, 1$", 0.5 * (g1**2 - g2**2)))
    fig, axes = plt.subplots(1, 3, figsize=(4.7, 1.75))
    for ax, (title, F) in zip(axes, cases):
        ax.contour(g1, g2, F, levels=12, colors=GREY, linewidths=0.5)
        ax.contour(g1, g2, F, levels=[0], colors=RED, linewidths=0.8)
        ax.set_title(title, fontsize=8.5, pad=4)
        ax.set_aspect("equal")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_xlabel(r"$w_1$")
    axes[0].set_ylabel(r"$w_2$")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.subplots_adjust(wspace=0.12)
    save(fig, "ch02-shapes")


def ch02_students():
    """Slices of the students' MSE through the minimiser: raw and centred."""
    from students import X, y, m, x1, x2
    Xc = np.column_stack([np.ones(m), x1 - x1.mean(), x2 - x2.mean()])
    fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.3))
    for ax, A, title in ((axes[0], X, "raw features"),
                         (axes[1], Xc, "centred features")):
        ws = np.linalg.lstsq(A, y, rcond=None)[0]
        a = np.linspace(ws[0] - 30, ws[0] + 30, 400)
        b = np.linspace(ws[2] - 30, ws[2] + 30, 400)
        A0, B0 = np.meshgrid(a, b)
        R = (A0[..., None] * A[:, 0] + ws[1] * A[:, 1]
             + B0[..., None] * A[:, 2] - y)
        L = (R**2).sum(-1) / (2 * m)
        L0 = L.min()
        ax.contour(A0, B0, L - L0, levels=[1, 5, 25, 100, 300],
                   colors=GREY, linewidths=0.6)
        ax.plot(ws[0], ws[2], "o", color=RED, ms=3)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=8.5, pad=4)
        ax.set_xlabel(r"$w_0$")
    axes[0].set_ylabel(r"$w_2$")
    fig.subplots_adjust(wspace=0.3)
    save(fig, "ch02-students")


def ch02_logistic():
    """Cross-entropy of the pass/fail students along a separating ray."""
    from students import X, y
    t = (y >= 65).astype(float)
    d = np.array([-13.0, 1.0, 1.5])
    c = np.linspace(0, 6, 300)
    ce = []
    for ci in c:
        z = X @ (ci * d)
        ce.append(np.mean(t * np.logaddexp(0, -z)
                          + (1 - t) * np.logaddexp(0, z)))
    ce = np.array(ce)
    reg = ce + 0.5 * 0.01 * c**2 * (d[1:] @ d[1:])
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(c, ce, color=BLUE, lw=1.3, label=r"cross-entropy")
    ax.plot(c, reg, color=ORANGE, lw=1.3,
            label=r"with penalty, $\rho = 0.01$")
    ax.set_xlabel(r"scale $s$ of $\mathbf{w} = s\,\mathbf{d}$")
    ax.set_ylabel("loss")
    ax.set_ylim(0, 0.92)
    ax.grid(lw=0.3)
    ax.legend(loc="upper right")
    save(fig, "ch02-logistic")


def _slice_path(A, n):
    """Exact line search on the slice w0 = w0*, weights (w1, w2), from (0, 8)."""
    from students import y, m
    H = A.T @ A / m
    ws = np.linalg.solve(H, A.T @ y / m)
    S = H[1:, 1:]
    c = ws[1:]
    v = np.array([0.0, 8.0])
    path = [v.copy()]
    for _ in range(n):
        gv = S @ (v - c)
        v = v - (gv @ gv) / (gv @ S @ gv) * gv
        path.append(v.copy())
    return np.array(path), S, c


def ch03_zigzag():
    """Exact line search on the students' slice: raw and standardised."""
    from students import X, m, x1, x2
    Xs = np.column_stack([np.ones(m), (x1 - x1.mean()) / x1.std(),
                          (x2 - x2.mean()) / x2.std()])
    fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.5))
    for ax, A, title, n in ((axes[0], X, "raw features", 51),
                            (axes[1], Xs, "standardised features", 8)):
        P, S, c = _slice_path(A, n)
        a = np.linspace(-1, 11, 400)
        b = np.linspace(-1, 11, 400)
        A0, B0 = np.meshgrid(a - c[0], b - c[1])
        Q = (S[0, 0] * A0**2 + 2 * S[0, 1] * A0 * B0 + S[1, 1] * B0**2) / 2
        ax.contour(A0 + c[0], B0 + c[1], Q, levels=[0.5, 2, 8, 32],
                   colors=GREY_RULE, linewidths=0.6)
        ax.plot(P[:, 0], P[:, 1], "-", color=BLUE, lw=0.8)
        ax.plot(P[:, 0], P[:, 1], "o", color=BLUE, ms=1.6)
        ax.plot(P[0, 0], P[0, 1], "o", color=ORANGE, ms=3)
        ax.plot(c[0], c[1], "o", color=RED, ms=3)
        ax.set_aspect("equal")
        ax.set_xlim(a[0], a[-1])
        ax.set_ylim(b[0], b[-1])
        ax.set_title(title, fontsize=8.5, pad=4)
        ax.set_xlabel(r"$w_1$")
    axes[0].set_ylabel(r"$w_2$")
    fig.subplots_adjust(wspace=0.3)
    save(fig, "ch03-zigzag")


def ch03_gaps():
    """Loss gap against steps for three step rules, raw features."""
    from students import X, y, m, H, w_star, loss, grad
    lam = np.linalg.eigvalsh(H)
    mu, be = lam[0], lam[-1]
    Ls = loss(w_star)
    rules = ((r"$\eta = 1/\beta$", lambda g: 1 / be, BLUE, "-"),
             (r"$\eta = 2/(\mu+\beta)$", lambda g: 2 / (mu + be), GREEN, "--"),
             ("exact line search", lambda g: (g @ g) / (g @ H @ g), ORANGE, "-"))
    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    T = 8000
    for label, rule, col, ls in rules:
        w = np.zeros(3)
        gap = [loss(w) - Ls]
        for _ in range(T):
            g = grad(w)
            w = w - rule(g) * g
            gap.append(loss(w) - Ls)
        ax.semilogy(np.arange(T + 1), gap, ls, color=col, lw=1.1, label=label)
    ax.axhline(0.01, color=GREY, lw=0.6, ls=":")
    ax.text(150, 0.013, "target 0.01", ha="left", va="bottom", fontsize=7.5)
    ax.set_xlabel(r"step $t$")
    ax.set_ylabel(r"$L(\mathbf{w}_t) - L^\star$")
    ax.set_xlim(0, T)
    ax.set_ylim(1e-4, 1e4)
    ax.grid(lw=0.3)
    ax.legend(loc="upper right")
    save(fig, "ch03-gaps")


def ch04_far():
    """Newton on f(w) = ln(1 + e^w) - 0.6 w from three starting points."""
    sig = lambda z: 1 / (1 + np.exp(-z))
    wmin = np.log(1.5)
    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    styles = ((1.0, BLUE, "o-"), (2.0, GREEN, "s--"), (2.5, RED, "^-"))
    for w0, col, st in styles:
        w, err = w0, []
        for _ in range(6):
            err.append(max(abs(w - wmin), 1e-16))
            if abs(w) > 50:
                break
            w = w - (sig(w) - 0.6) / (sig(w) * (1 - sig(w)))
        ax.semilogy(range(len(err)), err, st, color=col, lw=1.0, ms=3,
                    label=rf"$w_0 = {w0}$")
    ax.set_xlabel(r"step $t$")
    ax.set_ylabel(r"$|w_t - w^\star|$")
    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(1e-17, 1e3)
    ax.set_yticks([1e-15, 1e-10, 1e-5, 1, 1e3])
    ax.grid(lw=0.3)
    ax.legend(loc="lower left")
    save(fig, "ch04-far")


def ch04_ce():
    """Penalised cross-entropy of the students: Newton against gradient descent."""
    from students import X, y, m, H
    sig = lambda z: 1 / (1 + np.exp(-z))
    t_lab = (y >= 65).astype(float)
    rho = 0.01
    P = np.diag([0.0, 1.0, 1.0])
    L = lambda w: (np.mean(np.logaddexp(0, X @ w) - t_lab * (X @ w))
                   + 0.5 * rho * w[1:] @ w[1:])
    g = lambda w: X.T @ (sig(X @ w) - t_lab) / m + rho * P @ w

    def hess(w):
        p = sig(X @ w)
        return X.T @ np.diag(p * (1 - p)) @ X / m + rho * P

    w, path = np.zeros(3), []
    for _ in range(12):
        path.append(w.copy())
        w = w - np.linalg.solve(hess(w), g(w))
    Ls = L(w)
    newton = [L(v) - Ls for v in path[:7]]
    beta = np.linalg.eigvalsh(H)[-1] / 4 + rho
    T = 200000
    w, gd = np.zeros(3), []
    for _ in range(T + 1):
        gd.append(L(w) - Ls)
        w = w - g(w) / beta
    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    ax.loglog(np.arange(1, T + 2), np.maximum(gd, 1e-17), "-", color=BLUE,
              lw=1.1, label=r"gradient descent, $\eta = 1/\beta$")
    ax.loglog(np.arange(1, 8), newton, "o-", color=ORANGE, lw=1.0, ms=3,
              label="Newton")
    ax.set_xlabel(r"step $t + 1$")
    ax.set_ylabel(r"$L_\rho(\mathbf{w}_t) - L_\rho^\star$")
    ax.set_xlim(0.8, 3e5)
    ax.set_ylim(1e-15, 10)
    ax.set_yticks([1e-15, 1e-10, 1e-5, 1])
    ax.grid(lw=0.3)
    ax.legend(loc="lower left", bbox_to_anchor=(0.24, 0.0))
    save(fig, "ch04-ce")



def _avg_runs(fn, seeds=20):
    return np.mean([fn(s) for s in range(seeds)], axis=0)


def ch05_mean():
    """Robbins-Monro estimate of a mean with a constant and a 1/t step."""
    from ch05_stochastic import mean_run
    T = 2000
    t = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    ax.semilogx(t, mean_run(lambda s: 0.1, T), "-", color=ORANGE, lw=0.7,
                label=r"$\eta_t = 0.1$")
    ax.semilogx(t, mean_run(lambda s: 1 / s, T), "-", color=BLUE, lw=1.1,
                label=r"$\eta_t = 1/t$")
    ax.axhline(5.0, color=GREY, lw=0.6, ls="--")
    ax.set_xlabel(r"step $t$")
    ax.set_ylabel(r"estimate $w_t$")
    ax.set_xlim(1, T)
    ax.set_ylim(0, 8.5)
    ax.grid(lw=0.3)
    ax.legend(loc="lower right", ncol=2)
    save(fig, "ch05-mean")


def ch05_compare():
    """Full GD, LMS with constant steps and LMS with a decreasing step."""
    from ch05_stochastic import gd, sgd, dec
    E = 1000
    ep = np.arange(E + 1)
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    ax.semilogy(ep, np.maximum(gd(0.4, E), 1e-16), "-", color=GREY, lw=1.0,
                label=r"GD, $\eta = 0.4$")
    for eta, col, st in ((0.1, ORANGE, "-"), (0.02, GREEN, "-")):
        g = _avg_runs(lambda s: sgd(lambda t: eta, E, seed=s))
        ax.semilogy(ep, g, st, color=col, lw=1.0,
                    label=rf"LMS, $\eta = {eta}$")
    g = _avg_runs(lambda s: sgd(dec, E, seed=s))
    ax.semilogy(ep, g, "-", color=BLUE, lw=1.2,
                label=r"LMS, $\eta_t = c/(t + t_0)$")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, E)
    ax.set_ylim(1e-5, 3e3)
    ax.set_yticks([1e-4, 1e-2, 1, 1e2])
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$L(\mathbf{w}) - L^\star$")
    ax.grid(lw=0.3)
    ax.legend(loc="lower left", fontsize=7.5)
    save(fig, "ch05-compare")


def ch05_batch():
    """LMS with mini-batches of size 1, 2, 4, 8 and eta = 0.1."""
    from ch05_stochastic import sgd
    E = 1000
    ep = np.arange(E + 1)
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    for k, col in ((1, ORANGE), (2, RED), (4, GREEN), (8, BLUE)):
        g = _avg_runs(lambda s: sgd(lambda t: 0.1, E, k=k, seed=s))
        ax.semilogy(ep, g, "-", color=col, lw=1.0, label=rf"$k = {k}$")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, E)
    ax.set_ylim(3e-3, 3e3)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$L(\mathbf{w}) - L^\star$")
    ax.grid(lw=0.3)
    ax.legend(loc="lower left", ncol=2)
    save(fig, "ch05-batch")


def ch06_region():
    """Stable and complex-root regions of the heavy-ball mode recurrence."""
    g = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.fill_betweenx(g, 0, 2 * (1 + g), color=GREY_RULE, alpha=0.5, lw=0)
    ax.fill_betweenx(g, (1 - np.sqrt(g))**2, (1 + np.sqrt(g))**2,
                     color=BLUE, alpha=0.25, lw=0)
    ax.plot(2 * (1 + g), g, "-", color=GREY, lw=0.8)
    ax.plot((1 - np.sqrt(g))**2, g, "-", color=BLUE, lw=0.8)
    ax.plot((1 + np.sqrt(g))**2, g, "-", color=BLUE, lw=0.8)
    ax.text(1.35, 0.55, "complex\nroots", color=BLUE, ha="center",
            va="center", fontsize=8)
    ax.text(1.8, 0.02, "real", color=GREY, ha="center", fontsize=7)
    ax.text(0.3, 0.12, "real", color=GREY, ha="center", fontsize=8)
    ax.text(3.5, 0.3, "unstable", color=RED, ha="center", fontsize=8)
    ax.set_xlim(0, 4.2)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$\eta\lambda$")
    ax.set_ylabel(r"$\gamma$")
    save(fig, "ch06-region")


def ch06_path():
    """GD and heavy ball on the students' slice w0 = w0*, raw features."""
    from students import X, y, m
    H = X.T @ X / m
    ws = np.linalg.solve(H, X.T @ y / m)
    S, c = H[1:, 1:], ws[1:]
    lam = np.linalg.eigvalsh(S)
    mu, be = lam[0], lam[-1]
    eta_h = 4 / (np.sqrt(be) + np.sqrt(mu))**2
    gam = ((np.sqrt(be) - np.sqrt(mu)) / (np.sqrt(be) + np.sqrt(mu)))**2
    n = 30
    fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.5))
    for ax, (eta, gm, title) in zip(axes, ((2 / (mu + be), 0.0, "gradient descent"),
                                          (eta_h, gam, "heavy ball"))):
        v = np.array([0.0, 8.0])
        prev = v.copy()
        P = [v.copy()]
        for _ in range(n):
            v, prev = v - eta * S @ (v - c) + gm * (v - prev), v
            P.append(v.copy())
        P = np.array(P)
        a = np.linspace(-1, 11, 400)
        A0, B0 = np.meshgrid(a - c[0], a - c[1])
        Q = (S[0, 0] * A0**2 + 2 * S[0, 1] * A0 * B0 + S[1, 1] * B0**2) / 2
        ax.contour(A0 + c[0], B0 + c[1], Q, levels=[0.5, 2, 8, 32],
                   colors=GREY_RULE, linewidths=0.6)
        ax.plot(P[:, 0], P[:, 1], "-", color=BLUE, lw=0.8)
        ax.plot(P[:, 0], P[:, 1], "o", color=BLUE, ms=1.6)
        ax.plot(P[0, 0], P[0, 1], "o", color=ORANGE, ms=3)
        ax.plot(c[0], c[1], "o", color=RED, ms=3)
        ax.set_aspect("equal")
        ax.set_xlim(a[0], a[-1])
        ax.set_ylim(a[0], a[-1])
        ax.set_title(title, fontsize=8.5, pad=4)
        ax.set_xlabel(r"$w_1$")
    axes[0].set_ylabel(r"$w_2$")
    fig.subplots_adjust(wspace=0.3)
    save(fig, "ch06-path")


def ch06_gaps():
    """Loss gap of GD and heavy ball on the raw students."""
    from ch06_momentum import make, tune, hb, Lstar
    from students import X
    H, ws, L, g = make(X)
    lam = np.linalg.eigvalsh(H)
    mu, be = lam[0], lam[-1]
    T = 600
    t = np.arange(T + 1)
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    P = hb(g, 2 / (mu + be), 0.0, T)
    ax.semilogy(t, [L(p) - Lstar for p in P], "-", color=GREY, lw=1.0,
                label="GD")
    for f, col, lab in ((1.0, ORANGE, r"heavy ball, tuned for $\beta$"),
                        (1.1, BLUE, r"heavy ball, tuned for $1.1\beta$")):
        e, gm = tune(mu, f * be)
        P = hb(g, e, gm, T)
        ax.semilogy(t, [L(p) - Lstar for p in P], "-", color=col, lw=1.0,
                    label=lab)
    ax.axhline(0.01, color=GREY_RULE, lw=0.6, ls="--")
    ax.set_xlim(0, T)
    ax.set_ylim(1e-4, 3e6)
    ax.set_yticks([1e-4, 1e-2, 1, 1e2, 1e4, 1e6])
    ax.set_xlabel(r"step $t$")
    ax.set_ylabel(r"$L(\mathbf{w}_t) - L^\star$")
    ax.grid(lw=0.3)
    ax.legend(loc="upper right", fontsize=7.5)
    save(fig, "ch06-gaps")


def _bt(L, w, g, Lw, e0, tau=0.5, c1=1e-4):
    """Armijo backtracking along -g; returns (eta, loss evaluations)."""
    eta, k = e0, 0
    while True:
        k += 1
        if L(w - eta * g) <= Lw - c1 * eta * (g @ g):
            return eta, k
        eta *= tau


def ch07_phi():
    """phi on the students' first line, with Armijo and Wolfe intervals."""
    from students import H, loss, grad
    g = grad(np.zeros(3))
    gg, gHg, L0 = g @ g, g @ H @ g, loss(np.zeros(3))
    es = gg / gHg
    c1, c2 = 1e-4, 0.9
    e = np.linspace(0, 0.042, 400)
    phi = L0 - e * gg + 0.5 * e**2 * gHg
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(4.4, 2.9), sharex=True,
                                 gridspec_kw={"height_ratios": [3.2, 1],
                                              "hspace": 0.08})
    ax.plot(e, phi, "-", color=BLUE, lw=1.2, label=r"$\varphi(\eta)$")
    ax.plot(e, L0 + c1 * e * (-gg), "--", color=ORANGE, lw=1.0,
            label=r"Armijo line, $c_1 = 10^{-4}$")
    ax.plot(e, L0 - e * gg, ":", color=GREY, lw=1.0,
            label=r"tangent $\varphi(0) + \eta\varphi'(0)$")
    ea = 1 / 32
    ax.plot([ea], [L0 - ea * gg + 0.5 * ea**2 * gHg], "o", color=RED, ms=3.5)
    ax.annotate(r"accepted $\eta = 1/32$", (ea, 1579.6), (0.0165, 1850),
                fontsize=7.5, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.6))
    ax.set_ylim(0, 4400)
    ax.set_ylabel(r"$\varphi(\eta)$")
    ax.grid(lw=0.3)
    ax.legend(loc="upper center", fontsize=7.5, bbox_to_anchor=(0.43, 1.0))
    bars = (("Armijo", 0, 2 * (1 - c1) * es, ORANGE),
            ("Wolfe", (1 - c2) * es, 2 * (1 - c1) * es, GREEN),
            ("strong Wolfe", (1 - c2) * es, (1 + c2) * es, BLUE))
    for i, (lab, a, b, col) in enumerate(bars):
        yy = 2 - i
        bx.plot([a, b], [yy, yy], "-", color=col, lw=3, solid_capstyle="butt")
        bx.text(b + 0.0008, yy, lab, va="center", fontsize=7.5, color=col)
    bx.set_ylim(-0.7, 2.7)
    bx.set_yticks([])
    bx.set_xlim(0, 0.042)
    bx.set_xlabel(r"step $\eta$")
    save(fig, "ch07-phi")


def ch07_gaps():
    """Loss gap for fixed step, exact line search, and backtracking."""
    from students import H, w_star, loss, grad
    Ls = loss(w_star)
    T = 4200
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    for lab, kind, col, ls in ((r"fixed $\eta = 0.03$", "fix", GREY, "-"),
                               ("exact line search", "exact", ORANGE, "-"),
                               ("backtracking", "bt", BLUE, "--")):
        w = np.zeros(3)
        Lw, gap, eta = loss(w), [loss(w) - Ls], 0.5
        for _ in range(T):
            g = grad(w)
            if kind == "fix":
                eta = 0.03
            elif kind == "exact":
                eta = g @ g / (g @ H @ g)
            else:
                eta, _ = _bt(loss, w, g, Lw, 2 * eta)
            w = w - eta * g
            Lw = loss(w)
            gap.append(Lw - Ls)
        ax.semilogy(np.arange(T + 1), gap, ls, color=col, lw=1.1, label=lab)
    ax.axhline(0.01, color=GREY, lw=0.6, ls=":")
    ax.text(150, 0.013, "target 0.01", ha="left", va="bottom", fontsize=7.5)
    ax.set_xlabel(r"step $t$")
    ax.set_ylabel(r"$L(\mathbf{w}_t) - L^\star$")
    ax.set_xlim(0, T)
    ax.set_ylim(1e-3, 1e4)
    ax.grid(lw=0.3)
    ax.legend(loc="upper right")
    save(fig, "ch07-gaps")


def ch07_ce():
    """Accepted backtracking steps on the penalised cross-entropy."""
    from students import X, y, m, H
    t_lab = (y >= 65).astype(float)
    rho = 0.01
    sig = lambda z: 1 / (1 + np.exp(-z))
    L = lambda w: (np.mean(np.logaddexp(0, X @ w) - t_lab * (X @ w))
                   + 0.5 * rho * w[1:] @ w[1:])
    G = lambda w: X.T @ (sig(X @ w) - t_lab) / m + rho * np.r_[0, w[1:]]
    beta = np.linalg.eigvalsh(H)[-1] / 4 + rho
    T = 4000
    w, eta, etas = np.zeros(3), 0.5, []
    Lw = L(w)
    for _ in range(T):
        g = G(w)
        eta, _ = _bt(L, w, g, Lw, 1.0 if not etas else 2 * eta)
        w = w - eta * g
        Lw = L(w)
        etas.append(eta)
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    ax.semilogy(np.arange(T), etas, ".", color=BLUE, ms=1.2)
    ax.axhline(1 / beta, color=ORANGE, lw=0.9, ls="--")
    ax.text(T * 0.98, 1 / beta * 0.75, r"$1/\beta = 0.069$", ha="right",
            va="top", fontsize=7.5, color=ORANGE)
    ax.set_xlabel(r"step $t$")
    ax.set_ylabel(r"accepted $\eta_t$")
    ax.set_xlim(0, T)
    ax.set_ylim(0.02, 4)
    ax.grid(lw=0.3)
    save(fig, "ch07-ce")


def ch08_factor():
    """Per-mode factors of GD, heavy ball, and Nesterov for kappa = 25."""
    from ch06_momentum import tune
    from ch08_nesterov import sc_gamma, nag_roots
    mu, be = 1.0, 25.0
    lam = np.linspace(mu, be, 400)
    e, gm = tune(mu, be)
    gn = sc_gamma(be / mu)
    hbr = [abs(np.roots([1, -(1 + gm - e * l), gm])).max() for l in lam]
    nr = [abs(nag_roots(1 / be, gn, l)).max() for l in lam]
    fig, ax = plt.subplots(figsize=(3.8, 2.3))
    ax.plot(lam, np.abs(1 - 2 * lam / (mu + be)), color=GREY, lw=1.0,
            label=r"GD, $\eta = 2/(\mu+\beta)$")
    ax.plot(lam, hbr, color=ORANGE, lw=1.0, label="heavy ball, tuned")
    ax.plot(lam, nr, color=BLUE, lw=1.2,
            label=r"Nesterov, $\eta = 1/\beta$")
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 1.12)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.set_xlabel(r"eigenvalue $\lambda$")
    ax.set_ylabel(r"factor per step")
    ax.grid(lw=0.3)
    ax.legend(loc="upper center", fontsize=7.5)
    save(fig, "ch08-factor")


def ch08_gaps():
    """Loss gap of GD, heavy ball, and Nesterov on the raw students."""
    from ch06_momentum import make, tune
    from ch08_nesterov import nag, hb, sc_gamma, Lstar
    from students import X
    H, ws, L, g = make(X)
    lam = np.linalg.eigvalsh(H)
    mu, be = lam[0], lam[-1]
    T = 600
    t = np.arange(T + 1)
    e, gm = tune(mu, be)
    runs = [
        (hb(g, 2 / (mu + be), 0.0, T), GREY, "-", "GD"),
        (hb(g, e, gm, T), ORANGE, "-", "heavy ball"),
        (nag(g, 1 / be, sc_gamma(be / mu), T), BLUE, "-",
         r"Nesterov, constant $\gamma$"),
        (nag(g, 1 / be, None, T), GREEN, "-",
         r"Nesterov, $\gamma_t = (t-1)/(t+2)$"),
        (nag(g, 1 / be, None, T, L=L, restart=True), RED, "--",
         "Nesterov with restart"),
    ]
    fig, ax = plt.subplots(figsize=(4.7, 2.5))
    for P, col, ls, lab in runs:
        gap = np.maximum([L(p) - Lstar for p in P], 1e-16)
        ax.semilogy(t, gap, ls, color=col, lw=1.0, label=lab)
    ax.axhline(0.01, color=GREY_RULE, lw=0.6, ls="--")
    ax.set_xlim(0, T)
    ax.set_ylim(1e-12, 1e8)
    ax.set_yticks([1e-12, 1e-8, 1e-4, 1, 1e4, 1e8])
    ax.set_xlabel(r"step $t$")
    ax.set_ylabel(r"$L(\mathbf{w}_t) - L^\star$")
    ax.grid(lw=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7.5,
              ncol=2)
    save(fig, "ch08-gaps")


def ch09_net():
    """The 2-2-1 tanh network with bias nodes."""
    fig, ax = plt.subplots(figsize=(4.6, 2.35))
    ax.set_xlim(-0.2, 4.6)
    ax.set_ylim(-0.75, 2.75)
    ax.set_aspect("equal")
    ax.axis("off")
    inp = [(0.5, 2.2, "1"), (0.5, 1.2, "$x_1$"), (0.5, 0.2, "$x_2$")]
    hid = [(2.2, 2.2, "1"), (2.2, 1.2, "$a_1$"), (2.2, 0.2, "$a_2$")]
    out = (3.9, 1.2, r"$\hat y$")
    r = 0.22
    for xa, ya, _ in inp:
        for xb, yb, _ in hid[1:]:
            ax.annotate("", xy=(xb - r, yb), xytext=(xa + r, ya),
                        arrowprops=dict(arrowstyle="-|>", color=BLUE,
                                        lw=0.7, shrinkA=0, shrinkB=0))
    for xa, ya, _ in hid:
        ax.annotate("", xy=(out[0] - r, out[1]), xytext=(xa + r, ya),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=0.7,
                                    shrinkA=0, shrinkB=0))
    for (xa, ya, s), col in ([(p, GREY) for p in inp]
                             + [(p, BLUE) for p in hid] + [(out, GREEN)]):
        bias = s == "1"
        ax.add_patch(plt.Circle((xa, ya), r, fc="white",
                                ec=GREY_RULE if bias else col, lw=0.9))
        ax.text(xa, ya, s, ha="center", va="center", fontsize=9)
    ax.text(0.5, -0.5, r"input $\tilde{\mathbf{x}}$", ha="center")
    ax.text(2.2, -0.5, r"hidden $\tilde{\mathbf{a}}$", ha="center")
    ax.text(3.9, -0.5, "output", ha="center")
    ax.text(1.35, 2.55, r"$\mathbf{W}^{(1)}$", ha="center", color=BLUE)
    ax.text(3.05, 2.55, r"$\mathbf{w}^{(2)}$", ha="center", color=GREEN)
    ax.text(2.2, 1.62, r"$\tanh$", ha="center", fontsize=7.5, color=BLUE)
    ax.text(2.2, 0.62, r"$\tanh$", ha="center", fontsize=7.5, color=BLUE)
    save(fig, "ch09-net")


def ch09_curves():
    """Training loss of GD, heavy ball, and Nesterov on the 2-2-1 net."""
    from net221 import train, SY
    T = 2000
    lin = 0.289198 / SY ** 2
    runs = [(train(0.1, T)[2], GREY, "GD"),
            (train(0.1, T, 0.9)[2], ORANGE, r"heavy ball, $\gamma = 0.9$"),
            (train(0.1, T, 0.9, nesterov=True)[2], BLUE,
             r"Nesterov, $\gamma = 0.9$")]
    fig, ax = plt.subplots(figsize=(4.7, 2.5))
    t = np.arange(T + 1)
    for h, col, lab in runs:
        ax.semilogy(t, h, color=col, lw=1.0, label=lab)
    ax.axhline(lin, color=RED, lw=0.8, ls="--")
    ax.text(T, lin * 1.6, r"linear model $L^\star$", ha="right",
            va="bottom", color=RED, fontsize=7.5)
    ax.set_xlim(0, T)
    ax.set_ylim(1e-8, 2)
    ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
    ax.set_xlabel(r"step $t$")
    ax.set_ylabel(r"$L(\mathbf{w}_t)$, standard units")
    ax.grid(lw=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7.5,
              ncol=3)
    save(fig, "ch09-curves")


def ch09_swap():
    """Loss along the segment from a minimiser to its swapped copy."""
    from net221 import train, loss
    from ch09_backprop import swap
    V1, v2, _ = train(0.1, 2000)
    S1, s2 = swap(V1, v2)
    s = np.linspace(0, 1, 201)
    L = [loss((1 - u) * V1 + u * S1, (1 - u) * v2 + u * s2) for u in s]
    fig, ax = plt.subplots(figsize=(4.7, 2.0))
    ax.plot(s, L, color=BLUE, lw=1.0)
    ax.plot([0, 1], [L[0], L[-1]], "o", color=GREEN, ms=3.5, clip_on=False,
            zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel(r"position $s$ on the segment")
    ax.set_ylabel(r"loss")
    ax.grid(lw=0.3)
    save(fig, "ch09-swap")


def ch10_linear():
    """Loss gap of GD, delta-bar-delta, and Rprop on the raw linear model."""
    from ch10_perweight import linear_runs, LS
    r = linear_runs(1000)
    fig, ax = plt.subplots(figsize=(4.7, 2.4))
    t = np.arange(1001)
    for k, col, lab in (("GD", GREEN, r"GD, $\eta = 2/(\mu + \beta)$"),
                        ("DBD", ORANGE, "delta-bar-delta"),
                        ("Rprop+", BLUE, "Rprop")):
        h = np.array(r[k][1] if k == "GD" else r[k][2])
        ax.semilogy(t, np.maximum(h - LS, 1e-12), color=col, lw=1.0,
                    label=lab)
    ax.axhline(0.01, color=RED, lw=0.8, ls="--")
    ax.text(1000, 0.016, r"$L^\star + 0.01$", ha="right", va="bottom",
            color=RED, fontsize=7.5)
    ax.set_xlim(0, 1000)
    ax.set_ylim(1e-8, 1e4)
    ax.set_yticks([1e-8, 1e-4, 1, 1e4])
    ax.set_xlabel(r"step $t$")
    ax.set_ylabel(r"$L(\mathbf{w}_t) - L^\star$, points$^2$")
    ax.grid(lw=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7.5,
              ncol=3)
    save(fig, "ch10-linear")


def ch10_net():
    """Training loss of six methods on the 2-2-1 network, full batch."""
    from ch10_perweight import net_runs, LIN
    r = net_runs(400)
    style = {"GD": ("#9AA3AF", "-", "GD"), "HB": (ORANGE, "-", "heavy ball"),
             "NAG": (GREEN, "-", "Nesterov"),
             "DBD": (RED, "-", "delta-bar-delta"),
             "Rprop+": (BLUE, "-", "Rprop+"),
             "iRprop-": (BLUE, "--", r"iRprop$-$")}
    fig, ax = plt.subplots(figsize=(4.7, 2.6))
    t = np.arange(401)
    for k, h in r.items():
        col, ls, lab = style[k]
        ax.semilogy(t, h, color=col, ls=ls, lw=1.0, label=lab)
    ax.axhline(LIN, color=GREY, lw=0.6, ls=":")
    ax.text(400, LIN / 1.6, r"linear model $L^\star$", ha="right",
            va="top", color=GREY, fontsize=7.5)
    ax.set_xlim(0, 400)
    ax.set_ylim(1e-8, 2)
    ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
    ax.set_xlabel(r"step $t$")
    ax.set_ylabel(r"$L(\mathbf{w}_t)$, standard units")
    ax.grid(lw=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7.5,
              ncol=3)
    save(fig, "ch10-net")


def ch10_minibatch():
    """Rprop+ and SGD with mini-batches of two students."""
    from ch10_perweight import minibatch_runs, LIN
    rp, sgd, log = minibatch_runs()
    T = len(log)
    t = np.arange(T + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.1))
    a1.semilogy(t, rp[2], color=BLUE, lw=1.0, label="Rprop+")
    a1.semilogy(t, sgd[2], color=GREY, lw=0.8, label=r"SGD, $\eta = 0.1$")
    a1.axhline(LIN, color=RED, lw=0.8, ls="--")
    a1.text(T, LIN * 1.5, r"linear $L^\star$", ha="right", va="bottom",
            color=RED, fontsize=7.5)
    a1.set_xlim(0, T)
    a1.set_ylim(1e-4, 2)
    a1.set_xlabel(r"step $t$")
    a1.set_ylabel(r"$L(\mathbf{w}_t)$, full batch")
    a1.legend(loc="upper right", fontsize=7.5)
    D = np.array([d.max() for _, _, d, _ in log])
    a2.semilogy(np.arange(1, T + 1), D, color=BLUE, lw=1.0)
    a2.set_xlim(0, 200)
    a2.set_ylim(1e-7, 1)
    a2.set_yticks([1e-6, 1e-4, 1e-2, 1])
    a2.set_xlabel(r"step $t$")
    a2.set_ylabel(r"largest $\Delta_{j,t}$")
    for a in (a1, a2):
        a.grid(lw=0.3)
    fig.tight_layout(w_pad=1.5)
    save(fig, "ch10-minibatch")


def ch11_average():
    """Last iterate and half-tail average on the linear model, k = 1."""
    from ch11_averaging import lin_curve, Hc
    from ch05_stochastic import Xc, wc, y, m
    ts, last, avg = lin_curve()
    G = (Xc @ wc - y)[:, None] * Xc
    tr = np.trace(np.linalg.solve(Hc, G.T @ G / m))
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    ax.loglog(ts, last, color=GREY, lw=1.0, label="last iterate")
    ax.loglog(ts, avg, color=BLUE, lw=1.3, label="average over $[t/2, t]$")
    ax.loglog(ts, tr / ts, color=RED, lw=0.8, ls="--",
              label=r"$\mathrm{tr}(\mathbf{H}^{-1}\mathbf{\Sigma})/t$")
    ax.set_xlabel(r"step $t$")
    ax.set_ylabel(r"$L - L^\star$, points squared")
    ax.set_xlim(10, ts[-1])
    ax.legend(loc="lower left", fontsize=7.5)
    ax.grid(lw=0.3)
    save(fig, "ch11-average")


def ch11_batches():
    """Mini-batch SGD on the 2-2-1 network, and noise against loss."""
    from ch11_averaging import sgd, noise_path, LIN
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.2))
    cols = {1: RED, 2: ORANGE, 4: GREEN, 8: BLUE}
    for k in (1, 2, 4, 8):
        h = sgd(k, 0.1, 2000)[1]
        a1.semilogy(h, color=cols[k], lw=0.9, label=f"$k = {k}$")
    a1.axhline(LIN, color=GREY, lw=0.7, ls="--")
    a1.text(2000, LIN * 1.4, r"linear $L^\star$", ha="right", va="bottom",
            fontsize=7.5)
    a1.set_xlim(0, 2000)
    a1.set_xlabel(r"step $t$")
    a1.set_ylabel(r"$L(\mathbf{w}_t)$")
    a1.legend(loc="upper right", fontsize=7.5)
    L, S = noise_path()
    a2.loglog(L, S, ".", color=RED, ms=2)
    lg = np.array([1e-4, 1])
    a2.loglog(lg, 10 * lg, color=GREY, lw=0.7, ls="--")
    a2.text(1.5e-4, 0.3, r"$\sigma^2 = 10L$", fontsize=7.5)
    a2.set_xlabel(r"$L(\mathbf{w}_t)$")
    a2.set_ylabel(r"$\sigma^2(\mathbf{w}_t)$")
    for a in (a1, a2):
        a.grid(lw=0.3)
    fig.tight_layout(w_pad=1.5)
    save(fig, "ch11-batches")


def ch11_plateau():
    """Gradient descent from scaled copies of the starting weights."""
    from ch11_averaging import sgd, P0
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    cols = [BLUE, GREEN, ORANGE, RED]
    for s, c in zip((1, 0.1, 0.01, 0.001), cols):
        h = sgd(8, 0.1, 150, p=s * P0)[1]
        ax.plot(h, color=c, lw=1.0, label=rf"$s = {s:g}$")
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 0.7)
    ax.set_xlabel(r"step $t$")
    ax.set_ylabel(r"$L(\mathbf{w}_t)$")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(lw=0.3)
    save(fig, "ch11-plateau")


def ch11_slices():
    """Loss along the top and the flattest Hessian eigenvector."""
    from ch11_averaging import section_sharp, slices
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        ends = section_sharp()
    r, out = slices(ends)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.2), sharey=True)
    for k, c, lab in ((8, BLUE, "full batch"), (1, RED, "$k = 1$")):
        top, low, ev = out[k]
        a1.plot(r, top, color=c, lw=1.1,
                label=rf"{lab}, $\lambda_9 = {ev[-1]:.2f}$")
        a2.plot(r, low, color=c, lw=1.1)
    a1.set_xlabel(r"step $s$ along $\mathbf{q}_9$")
    a2.set_xlabel(r"step $s$ along $\mathbf{q}_1$")
    a1.set_ylabel(r"$L$")
    a1.set_ylim(0, 1.0)
    fig.legend(loc="upper center", ncol=2, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.08))
    for a in (a1, a2):
        a.set_xlim(-1, 1)
        a.grid(lw=0.3)
    fig.tight_layout(w_pad=1.2)
    save(fig, "ch11-slices")


def ch12_linear():
    """Loss gap of GD, delta-bar-delta, Rprop+ and AdaGrad, raw features."""
    from ch12_adagrad import linear_runs, minute_runs, LS
    r = linear_runs(2500)
    mins = minute_runs(2500)[0]
    fig, ax = plt.subplots(figsize=(4.7, 2.5))
    t = np.arange(2501)
    for h, col, ls, lab in (
            (r["GD 0.03"][2], GREEN, "-", r"GD, $\eta = 0.03$"),
            (r["DBD"][2], ORANGE, "-", "delta-bar-delta"),
            (r["Rprop+"][2], GREY, "-", "Rprop+"),
            (r["AdaGrad 10"][2], BLUE, "-", r"AdaGrad, $\eta = 10$"),
            (mins["AdaGrad 10"][2], BLUE, "--", "AdaGrad, minutes")):
        ax.semilogy(t, np.maximum(np.array(h) - LS, 1e-12), color=col,
                    ls=ls, lw=1.0, label=lab)
    ax.axhline(0.01, color=RED, lw=0.8, ls=":")
    ax.text(2500, 0.016, r"$L^\star + 0.01$", ha="right", va="bottom",
            color=RED, fontsize=7.5)
    ax.set_xlim(0, 2500)
    ax.set_ylim(1e-8, 1e4)
    ax.set_yticks([1e-8, 1e-4, 1, 1e4])
    ax.set_xlabel(r"step $t$")
    ax.set_ylabel(r"$L(\mathbf{w}_t) - L^\star$, points$^2$")
    ax.grid(lw=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7.5,
              ncol=3)
    save(fig, "ch12-linear")


def ch12_net():
    """AdaGrad and earlier methods on the 2-2-1 network."""
    from ch12_adagrad import net_runs, minibatch_runs, LIN
    full = net_runs(400)
    mini = minibatch_runs()[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.5), sharey=True)
    st = {"GD": ("#9AA3AF", "-", "GD / SGD"), "SGD": ("#9AA3AF", "-", None),
          "HB": (ORANGE, "-", "heavy ball"), "Rprop+": (GREEN, "-", "Rprop+"),
          "AdaGrad 0.1": (BLUE, ":", r"AdaGrad, $\eta = 0.1$"),
          "AdaGrad 0.5": (BLUE, "-", r"AdaGrad, $\eta = 0.5$"),
          "AdaGrad": (BLUE, "-", None)}
    for k, h in full.items():
        if k in st:
            col, ls, lab = st[k]
            a1.semilogy(np.arange(len(h)), h, color=col, ls=ls, lw=1.0,
                        label=lab)
    for k, h in mini.items():
        col, ls, _ = st[k]
        a2.semilogy(np.arange(len(h)), h, color=col, ls=ls, lw=1.0)
    a1.set_xlim(0, 400)
    a2.set_xlim(0, 2000)
    a2.set_xticks([0, 1000, 2000])
    a1.set_title("full batch")
    a2.set_title("mini-batches of two")
    for a in (a1, a2):
        a.axhline(LIN, color=RED, lw=0.8, ls="--")
        a.set_ylim(1e-12, 2)
        a.set_yticks([1e-12, 1e-8, 1e-4, 1])
        a.set_xlabel(r"step $t$")
        a.grid(lw=0.3)
    a1.set_ylabel(r"$L(\mathbf{w}_t)$, standard units")
    fig.legend(*a1.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.02), fontsize=7.5, ncol=3)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch12-net")


def ch12_rates():
    """Effective rates of the nine network weights under AdaGrad."""
    from ch12_adagrad import rate_log
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.2), sharey=True)
    for a, R, title in ((a1, rate_log(), "full batch"),
                        (a2, rate_log(b=2), "mini-batches of two")):
        t = np.arange(1, len(R) + 1)
        for j in range(R.shape[1]):
            a.loglog(t, R[:, j], color=BLUE if j < 6 else ORANGE, lw=0.8)
        a.axhline(0.5, color=RED, lw=0.8, ls="--")
        a.set_title(title)
        a.set_xlim(1, len(R))
        a.set_ylim(0.1, 40)
        a.set_xlabel(r"step $t$")
        a.grid(lw=0.3)
    a1.text(1.5, 0.43, r"$\eta = 0.5$", color=RED, va="top", fontsize=7.5)
    a1.set_ylabel(r"$\eta/(\sqrt{h_{j,t}} + \zeta)$")
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch12-rates")


def ch13_linear():
    """Loss gaps and the effective rate of w_2 on the raw linear model."""
    from ch13_rmsprop import linear_runs, linear_rates, LS
    r = linear_runs()
    ra, rr = linear_rates()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.4))
    t = np.arange(len(r["GD 0.03"][2]))
    for k, col, ls, lab in (
            ("GD 0.03", GREEN, "-", r"GD, $\eta = 0.03$"),
            ("AdaGrad 10", BLUE, "-", r"AdaGrad, $\eta = 10$"),
            ("RMSProp 0.01", ORANGE, "-", r"RMSProp, $\eta = 0.01$"),
            ("RMSProp 0.003", ORANGE, ":", r"RMSProp, $\eta = 0.003$"),
            ("AdaDelta", GREY, "-", "AdaDelta")):
        a1.loglog(t[1:], np.maximum(np.array(r[k][2][1:]) - LS, 1e-12),
                  color=col, ls=ls, lw=1.0, label=lab)
    a1.axhline(0.01, color=RED, lw=0.8, ls="--")
    a1.set_ylim(1e-6, 1e4)
    a1.set_yticks([1e-6, 1e-3, 1, 1e3])
    a1.set_ylabel(r"$L(\mathbf{w}_t) - L^\star$, points$^2$")
    a1.set_title("loss gap")
    s = np.arange(1, len(ra) + 1)
    a2.loglog(s, ra, color=BLUE, lw=1.0)
    a2.loglog(s, rr, color=ORANGE, lw=1.0)
    a2.set_ylim(1e-5, 1)
    a2.set_title(r"effective rate of $w_2$")
    for a in (a1, a2):
        a.set_xlim(1, len(ra))
        a.set_xticks([1, 100, 10000])
        a.set_xlabel(r"step $t$")
        a.grid(lw=0.3)
    fig.legend(*a1.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.02), fontsize=7.5, ncol=3)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch13-linear")


def ch13_net():
    """RMSProp, AdaDelta and earlier methods on the 2-2-1 network."""
    from ch13_rmsprop import net_runs, minibatch_runs, LIN
    full = net_runs()
    mini = minibatch_runs()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.5), sharey=True)
    st = {"GD": ("#9AA3AF", "-", "GD / SGD"), "SGD": ("#9AA3AF", "-", None),
          "HB": ("#6B4C9A", "-", "heavy ball"),
          "Rprop+": (GREEN, "-", "Rprop+"),
          "AdaGrad 0.01": (BLUE, ":", r"AdaGrad, $\eta = 0.01$"),
          "AdaGrad 0.5": (BLUE, "-", r"AdaGrad, $\eta = 0.5$"),
          "AdaGrad": (BLUE, "-", None),
          "RMSProp 0.01": (ORANGE, "-", r"RMSProp, $\eta = 0.01$"),
          "RMSProp": (ORANGE, "-", None),
          "AdaDelta": (GREY, "-", "AdaDelta")}
    for a, runs in ((a1, full), (a2, mini)):
        for k, h in runs.items():
            col, ls, lab = st[k]
            a.semilogy(np.arange(len(h)), h, color=col, ls=ls, lw=1.0,
                       label=lab if a is a1 else None)
    a1.set_title("full batch")
    a2.set_title("mini-batches of two")
    for a in (a1, a2):
        a.axhline(LIN, color=RED, lw=0.8, ls="--")
        a.set_xlim(0, 2000)
        a.set_xticks([0, 1000, 2000])
        a.set_ylim(1e-12, 2)
        a.set_yticks([1e-12, 1e-8, 1e-4, 1])
        a.set_xlabel(r"step $t$")
        a.grid(lw=0.3)
    a1.set_ylabel(r"$L(\mathbf{w}_t)$, standard units")
    fig.legend(*a1.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.02), fontsize=7.5, ncol=3)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch13-net")


def ch13_rates():
    """Effective rates of AdaGrad and RMSProp, eta = 0.01, mini-batches."""
    from ch13_rmsprop import net_rates
    ra, rr = net_rates()[:2]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.2), sharey=True)
    for a, R, title in ((a1, ra, "AdaGrad"), (a2, rr, "RMSProp")):
        t = np.arange(1, len(R) + 1)
        for j in range(R.shape[1]):
            a.loglog(t, R[:, j], color=BLUE if j < 6 else ORANGE, lw=0.8)
        a.set_title(title)
        a.set_xlim(1, len(R))
        a.set_ylim(1e-4, 10)
        a.set_xlabel(r"step $t$")
        a.grid(lw=0.3)
    a1.set_ylabel("effective rate")
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch13-rates")


def ch14_bias():
    """Move of uncorrected Adam for a constant gradient; net with/without."""
    from ch14_adam import bias_runs, LIN
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.3))
    t = np.arange(0, 5000)
    for om2, ls in ((0.999, "-"), (0.99, ":")):
        a1.semilogx(t + 1, (1 - 0.9 ** (t + 1)) / np.sqrt(1 - om2 ** (t + 1)),
                    color=ORANGE, ls=ls, lw=1.0,
                    label=rf"no correction, $\omega_2 = {om2}$")
    a1.semilogx(t + 1, np.ones(len(t)), color=BLUE, lw=1.0,
                label="with correction")
    a1.set_xlim(1, 5000)
    a1.set_ylim(0, 7.5)
    a1.set_xlabel(r"step $t + 1$")
    a1.set_ylabel(r"move$/\eta$")
    a1.set_title("constant derivative")
    a1.legend(loc="upper right", fontsize=7, handlelength=1.6)
    br = bias_runs()
    for c, col, lab in ((True, BLUE, "with correction"),
                        (False, ORANGE, "no correction")):
        h = br[c][0]
        a2.semilogy(np.arange(len(h)), h, color=col, lw=1.0, label=lab)
    a2.axhline(LIN, color=RED, lw=0.8, ls="--")
    a2.set_xlim(0, 300)
    a2.set_ylim(1e-8, 2)
    a2.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
    a2.set_xlabel(r"step $t$")
    a2.set_ylabel(r"$L(\mathbf{w}_t)$")
    a2.set_title(r"network, $\eta = 0.03$")
    a2.legend(loc="lower left", fontsize=7, handlelength=1.6)
    for a in (a1, a2):
        a.grid(lw=0.3)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch14-bias")


def ch14_linear():
    """Loss gaps of Adam, AdaMax, Nadam and earlier methods, raw linear."""
    from ch14_adam import linear_runs, LS
    r = linear_runs()
    fig, a = plt.subplots(figsize=(4.6, 2.5))
    t = np.arange(len(r["GD 0.03"][2]))
    for k, col, ls, lab in (
            ("GD 0.03", GREEN, "-", r"GD, $\eta = 0.03$"),
            ("AdaGrad 10", BLUE, "-", r"AdaGrad, $\eta = 10$"),
            ("RMSProp 0.01", GREY, "-", r"RMSProp, $\eta = 0.01$"),
            ("adam 0.1", ORANGE, "-", r"Adam, $\eta = 0.1$"),
            ("adamax 0.1", "#6B4C9A", "-", r"AdaMax, $\eta = 0.1$"),
            ("nadam 0.1", RED, ":", r"Nadam, $\eta = 0.1$")):
        a.loglog(t[1:], np.maximum(np.array(r[k][2][1:]) - LS, 1e-12),
                 color=col, ls=ls, lw=0.9, label=lab)
    a.axhline(0.01, color=RED, lw=0.6, ls="--")
    a.set_xlim(1, len(t))
    a.set_ylim(1e-12, 1e4)
    a.set_yticks([1e-12, 1e-8, 1e-4, 1, 1e4])
    a.set_xlabel(r"step $t$")
    a.set_ylabel(r"$L(\mathbf{w}_t) - L^\star$, points$^2$")
    a.grid(lw=0.3)
    fig.legend(*a.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.02), fontsize=7.5, ncol=3)
    fig.tight_layout()
    save(fig, "ch14-linear")


def ch14_net():
    """Adam, AdaMax, Nadam and earlier methods on the 2-2-1 network."""
    from ch14_adam import net_runs, minibatch_runs, LIN
    full = net_runs()
    mini = minibatch_runs()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.5), sharey=True)
    st = {"GD": None, "SGD": None, "Adam 0.01": None,
          "HB": ("#9AA3AF", "-", "heavy ball"),
          "NAG": (GREEN, "-", "Nesterov"),
          "RMSProp": (GREY, "-", "RMSProp"),
          "adam 0.03": (ORANGE, "-", "Adam"), "adam": (ORANGE, "-", None),
          "adamax 0.03": ("#6B4C9A", "-", "AdaMax"),
          "adamax": ("#6B4C9A", "-", None),
          "nadam 0.03": (BLUE, ":", "Nadam"), "nadam": (BLUE, ":", None)}
    for a, runs in ((a1, full), (a2, mini)):
        for k, h in runs.items():
            if st[k] is None:
                continue
            col, ls, lab = st[k]
            a.semilogy(np.arange(len(h)), np.maximum(h, 1e-16), color=col,
                       ls=ls, lw=1.0, label=lab if a is a1 else None)
    a1.set_title("full batch")
    a2.set_title("mini-batches of two")
    for a in (a1, a2):
        a.axhline(LIN, color=RED, lw=0.8, ls="--")
        a.set_xlim(0, 2000)
        a.set_xticks([0, 1000, 2000])
        a.set_ylim(1e-16, 2)
        a.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1])
        a.set_xlabel(r"step $t$")
        a.grid(lw=0.3)
    a1.set_ylabel(r"$L(\mathbf{w}_t)$, standard units")
    fig.legend(*a1.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.02), fontsize=7.5, ncol=3)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch14-net")


def ch15_shapes():
    """SGDR restarts; warm-up with cosine and with inverse square root."""
    from ch15_schedules import sgdr, cosine
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.1))
    t = np.arange(350)
    f = sgdr(1.0, 50)
    a1.plot(t, [f(s) for s in t], color=BLUE, lw=1.0)
    for r in (50, 150):
        a1.axvline(r, color=GREY_RULE, lw=0.6, ls="--")
    a1.set_title(r"SGDR, $T_0 = 50$, $T_{\mathrm{mult}} = 2$")
    a1.set_xlabel(r"step $t$")
    a1.set_ylabel(r"$\eta_t / \eta_{\max}$")
    a1.set_xticks([0, 50, 150, 350])
    s = np.arange(1, 1201)
    tw = 100
    c = cosine(1.0, 1200, tw=tw)
    a2.plot(s, [c(u - 1) for u in s], color=ORANGE, lw=1.0,
            label="warm-up + cosine")
    a2.plot(s, np.minimum(s ** -0.5, s * tw ** -1.5) * tw ** 0.5,
            color=GREEN, lw=1.0, label="warm-up + $1/\\sqrt{s}$")
    a2.set_title(r"warm-up, $t_w = 100$")
    a2.set_xlabel(r"step $t$")
    a2.set_xticks([100, 600, 1200])
    fig.legend(*a2.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.02), fontsize=7.5, ncol=2)
    for a in (a1, a2):
        a.set_ylim(0, 1.08)
        a.grid(lw=0.3)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch15-shapes")


def ch15_linear():
    """Six schedules on the centred linear model, batches of two."""
    from ch15_schedules import LIN_SCHED, lin_run, T_LIN
    st = {"constant 0.4": (GREY, "-"), "constant 0.04": ("#9AA3AF", "--"),
          "step": (BLUE, "-"), "exponential": (GREEN, "-"),
          "1/t": ("#6B4C9A", "-"), "cosine": (ORANGE, "-")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.5))
    t = np.arange(T_LIN + 1)
    for k, f in LIN_SCHED.items():
        col, ls = st[k]
        a1.semilogy(t[:-1], [f(s) for s in t[:-1]], color=col, ls=ls,
                    lw=1.0, label=k)
        G = np.mean([lin_run(f, seed=s) for s in range(500)], axis=0)
        a2.semilogy(t, G, color=col, ls=ls, lw=1.0)
    a1.set_ylabel(r"$\eta_t$")
    a1.set_ylim(1e-6, 1)
    a1.set_title("schedules")
    a2.set_ylabel(r"mean of $L(\mathbf{w}_t) - L^\star$")
    a2.set_ylim(1e-3, 1e4)
    a2.set_title("loss gap, 500 runs")
    for a in (a1, a2):
        a.set_xlabel(r"step $t$")
        a.set_xticks([0, 100, 200, 300, 400])
        a.grid(lw=0.3)
    fig.legend(*a1.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.02), fontsize=7.5, ncol=3)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch15-linear")


def ch15_net():
    """Heavy ball and Adam with a large rate, with and without warm-up."""
    from ch15_schedules import net_run, variants, LIN
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.5), sharey=True)
    st = {"constant": (GREY, "-"), "warm-up": (BLUE, "--"),
          "warm-up + cosine": (ORANGE, "-")}
    for a, meth, e in ((a1, "HB", 1.5), (a2, "Adam", 1.0)):
        for k, f in variants(e).items():
            if k not in st:
                continue
            h = net_run(meth, f)
            col, ls = st[k]
            a.semilogy(np.arange(len(h)), np.clip(h, 1e-34, 1e3), color=col,
                       ls=ls, lw=1.0, label=k if a is a1 else None)
        a.axhline(LIN, color=RED, lw=0.8, ls="--")
        a.set_xlim(0, 2000)
        a.set_xticks([0, 1000, 2000])
        a.set_ylim(1e-34, 1e3)
        a.set_yticks([1e-32, 1e-24, 1e-16, 1e-8, 1])
        a.set_xlabel(r"step $t$")
        a.grid(lw=0.3)
    a1.set_title(r"heavy ball, $\eta = 1.5$")
    a2.set_title(r"Adam, $\eta = 1$")
    a1.set_ylabel(r"$L(\mathbf{w}_t)$, standard units")
    fig.legend(*a1.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.02), fontsize=7.5, ncol=3)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch15-net")


def ch15_range():
    """Learning-rate range test on the 2-2-1 network, batches of two."""
    from ch15_schedules import range_test, P0, loss_flat
    fig, a = plt.subplots(figsize=(3.6, 2.4))
    for meth, lo, hi, col in (("HB", 1e-3, 10.0, BLUE),
                              ("Adam", 1e-4, 3.0, ORANGE)):
        e, h = range_test(lo, hi, 400, meth=meth)
        a.loglog(e, np.clip(h, 1e-4, 1e2), color=col, lw=1.0,
                 label="heavy ball" if meth == "HB" else "Adam")
        i = np.argmin(np.where(np.isfinite(h), h, np.inf))
        a.plot(e[i], h[i], "o", ms=3.5, color=col)
    a.axhline(loss_flat(P0), color=GREY, lw=0.6, ls=":")
    a.set_xlabel(r"learning rate $\eta$")
    a.set_ylabel(r"$L(\mathbf{w}_t)$, standard units")
    a.set_ylim(1e-4, 1e2)
    a.set_xlim(1e-4, 10)
    a.grid(lw=0.3)
    a.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    save(fig, "ch15-range")


def ch16_counter():
    """Iterates on the two online counterexamples of Reddi et al."""
    from ch16_amsgrad import thm_runs, syn_runs
    R, S = thm_runs(), syn_runs()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.5), sharey=True)
    st = {"sgd": (GREY, "-", "SGD"), "SGD": (GREY, "-", None),
          "rmsprop": (ORANGE, "-", "RMSProp"),
          "RMSProp": (ORANGE, "-", None),
          "Adam": (RED, ":", "Adam"),
          "amsgrad": (BLUE, "-", "AMSGrad"), "AMSGrad": (BLUE, "-", None)}
    for k, run in R.items():
        col, ls, lab = st[k]
        a1.plot(np.arange(len(run[0])), run[0], color=col, ls=ls, lw=0.9,
                label=lab)
    for k, run in S.items():
        col, ls, lab = st[k]
        cm = run[0][:-1].reshape(-1, 101).mean(1)
        a2.semilogx(np.arange(1, len(cm) + 1), cm, color=col, ls=ls,
                    lw=1.1, label=lab)
    a1.set_title(r"$C = 3$, period 3")
    a1.set_xlabel(r"step $t$")
    a1.set_xlim(0, 300)
    a1.set_xticks([0, 100, 200, 300])
    a1.set_ylabel(r"iterate $x_t$")
    a2.set_title(r"$C = 1010$, period 101")
    a2.set_xlabel("period $k$ (mean over it)")
    a2.set_xlim(1, 2000)
    for a in (a1, a2):
        a.set_ylim(-1.1, 1.1)
        a.set_yticks([-1, 0, 1])
        a.grid(lw=0.3)
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center",
               bbox_to_anchor=(0.5, 0.02), fontsize=7.5, ncol=4)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch16-counter")


def ch16_net():
    """Adam and AMSGrad on the 2-2-1 network."""
    from ch16_amsgrad import net_pair, LIN
    full, mini = net_pair(0.01), net_pair(0.03, b=2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.4), sharey=True)
    st = {"Adam": (ORANGE, "-"), "AMSGrad": (BLUE, ":")}
    for a, runs in ((a1, full), (a2, mini)):
        for k, (p, h, log) in runs.items():
            col, ls = st[k]
            a.semilogy(np.arange(len(h)), h, color=col, ls=ls, lw=1.1,
                       label=k if a is a1 else None)
    a1.set_title(r"full batch, $\eta = 0.01$")
    a2.set_title(r"mini-batches of two, $\eta = 0.03$")
    for a in (a1, a2):
        a.axhline(LIN, color=RED, lw=0.8, ls="--")
        a.set_xlim(0, 2000)
        a.set_xticks([0, 1000, 2000])
        a.set_ylim(1e-6, 2)
        a.set_yticks([1e-6, 1e-4, 1e-2, 1])
        a.set_xlabel(r"step $t$")
        a.grid(lw=0.3)
    a1.set_ylabel(r"$L(\mathbf{w}_t)$, standard units")
    fig.legend(*a1.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.02), fontsize=7.5, ncol=2)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch16-net")


def ch17_decay():
    """Per-weight decay rate of Adam+L2 on the 2-2-1 network."""
    from ch17_adamw import eff_decay
    ed = eff_decay()
    fig, a = plt.subplots(figsize=(4.6, 2.4))
    pick = [(1, r"$w^{(1)}_{11}$", BLUE, "-"),
            (5, r"$w^{(1)}_{22}$", GREEN, "-."),
            (7, r"$w^{(2)}_{1}$", ORANGE, ":")]
    t = np.arange(len(ed))
    for j, lab, col, ls in pick:
        a.semilogy(t, ed[:, j], color=col, ls=ls, lw=1.1, label=lab)
    a.axhline(0.03 * 0.01, color=RED, lw=1.0, ls="--",
              label=r"AdamW, $\eta\rho$")
    a.set_xlim(0, 2000)
    a.set_xticks([0, 500, 1000, 1500, 2000])
    a.set_ylim(1e-4, 0.2)
    a.set_xlabel(r"step $t$")
    a.set_ylabel("decay rate per step")
    a.grid(lw=0.3)
    a.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7.5,
             frameon=False)
    fig.tight_layout()
    save(fig, "ch17-decay")


def ch17_net():
    """Sweep of rho for Adam+L2 and AdamW on the 2-2-1 network."""
    from ch17_adamw import net_sweep
    rows = net_sweep()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.4))
    st = {False: (ORANGE, "o", "Adam+L2"), True: (BLUE, "s", "AdamW")}
    for dec in (False, True):
        col, mk, lab = st[dec]
        r = np.array([[q[0], q[2], q[4]] for q in rows if q[1] == dec])
        a1.loglog(r[:, 0], r[:, 1], color=col, marker=mk, ms=3.5, lw=1.0,
                  label=lab)
        a2.loglog(r[:, 0], r[:, 2], color=col, marker=mk, ms=3.5, lw=1.0)
    rr = np.array([0.6, 12.0])
    a2.loglog(rr, 1 / rr, color=RED, lw=0.9, ls="--",
              label=r"$1/\rho$")
    a1.set_ylabel(r"final $L$, standard units")
    a2.set_ylabel(r"largest $|w_j|$")
    for a in (a1, a2):
        a.set_xlabel(r"$\rho$")
        a.set_xlim(5e-4, 20)
        a.set_xticks([1e-3, 1e-2, 1e-1, 1, 10])
        a.grid(lw=0.3)
    a1.set_ylim(1e-4, 1)
    a2.set_ylim(0.05, 3)
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center",
               bbox_to_anchor=(0.5, 0.02), fontsize=7.5, ncol=3)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch17-net")


def ch18_ratio():
    """Layer ratios ||w||/||g|| and the noise scale along GD on the net."""
    from ch18_lars import ratio_path, noise_scale
    from net221 import init, flat, grad_flat
    R = ratio_path(0.1, 3000)
    p, B = flat(*init()), []
    for _ in range(3001):
        B.append(noise_scale(p))
        p = p - 0.1 * grad_flat(p)
    t = np.arange(1, 3002)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.3))
    a1.loglog(t, R[:, 0], color=BLUE, lw=1.0, label=r"layer 1, $\mathbf{W}^{(1)}$")
    a1.loglog(t, R[:, 1], color=ORANGE, lw=1.0, label=r"layer 2, $\boldsymbol{w}^{(2)}$")
    a1.set_ylabel(r"$\|\boldsymbol{w}^{(l)}\| \,/\, \|\boldsymbol{g}^{(l)}\|$")
    a1.set_ylim(0.3, 5e3)
    a2.loglog(t, B, color=GREEN, lw=1.0, label="noise scale")
    a2.axhline(8, color=RED, lw=0.9, ls="--", label=r"$m = 8$")
    a2.set_ylabel(r"$\mathrm{tr}\,\Sigma \,/\, \|\boldsymbol{g}\|^2$")
    a2.set_ylim(1, 5e3)
    for a in (a1, a2):
        a.set_xlabel("step $t + 1$")
        a.set_xlim(1, 3001)
        a.grid(lw=0.3)
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center",
               bbox_to_anchor=(0.5, 0.02), fontsize=7.5, ncol=4,
               columnspacing=1.0, handlelength=1.6)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch18-ratio")


def ch18_batch():
    """Losses per epoch of four methods with batches of 8 and of 1."""
    from ch18_lars import batch_table
    base = {"SGDM": 0.3, "LARS": 0.05, "AdamW": 0.1, "LAMB": 0.03}
    rows = {k: res for k, _, res in batch_table(base)}
    st = {"SGDM": (BLUE, "-"), "LARS": (ORANGE, "-"),
          "AdamW": (GREEN, "--"), "LAMB": (RED, "--")}
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.3), sharey=True)
    for a, k in zip(axs, (8, 1)):
        for name, (col, ls) in st.items():
            h = np.array(rows[k][name][3])
            ep = np.arange(len(h)) * k / 8
            a.semilogy(ep, h, color=col, ls=ls, lw=0.9, label=name)
        a.axhline(0.289198 / 123.4375, color=GREY, lw=0.7, ls=":",
                  label=r"linear $L^\star$")
        a.set_title(f"$k = {k}$")
        a.set_xlabel("epoch")
        a.set_xlim(0, 400)
        a.set_ylim(1e-6, 1)
        a.grid(lw=0.3)
    axs[0].set_ylabel("$L$, standard units")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               fontsize=7.5, ncol=5, columnspacing=1.0, handlelength=1.8)
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch18-batch")


def ch19_radam():
    """Variance of 1/sqrt(rhat_t) and the rectification term of RAdam."""
    from ch19_stability import q_t, q_inf, nu_t, var_approx, mc_var
    ts = sorted({int(round(v)) for v in np.geomspace(5, 2000, 18)})
    mc = mc_var(set(ts), n=100000)
    tt = np.arange(5, 2001)
    q = q_t(tt)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.3))
    a1.loglog(tt, var_approx(q), color=BLUE, lw=1.0,
              label="approximation")
    a1.loglog(ts, [mc[t] for t in ts], "o", ms=2.8, color=ORANGE,
              label="simulation")
    a1.loglog(ts, [nu_t(q_t(t), q_inf()) ** 2 * mc[t] for t in ts], "s",
              ms=2.6, color=GREEN, label="rectified (sim.)")
    a1.set_xlabel("step $t$")
    a1.set_ylabel(r"variance of $1/\sqrt{\hat r_t}$")
    a1.set_xlim(4, 2500)
    a1.set_ylim(5e-5, 2)
    for om2, col, ls in ((0.999, BLUE, "-"), (0.99, ORANGE, "--")):
        t2 = np.arange(1, 20001)
        q2 = q_t(t2, om2)
        nu = np.where(q2 > 4, nu_t(np.maximum(q2, 4.0001), q_inf(om2)),
                      np.nan)
        a2.semilogx(t2, nu, color=col, ls=ls, lw=1.0,
                    label=rf"$\omega_2 = {om2}$")
    a2.set_xlabel("step $t$")
    a2.set_ylabel(r"$\nu_t$")
    a2.set_xlim(1, 20000)
    a2.set_ylim(0, 1.05)
    for a in (a1, a2):
        a.grid(lw=0.3)
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center",
               bbox_to_anchor=(0.5, 0.02), fontsize=7.5, ncol=3,
               columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch19-radam")


def ch19_lookahead():
    """GD and Lookahead(GD) on the standardised linear model, eta = 1.3."""
    from ch19_stability import lookahead_hand, lin_parts
    from net221 import ys, m
    Hs, bs, rows, gd = lookahead_hand()
    Hs, bs = Hs[1:, 1:], bs[1:]
    ws = np.linalg.solve(Hs, bs)
    A, B = np.meshgrid(np.linspace(-0.15, 1.35, 300),
                       np.linspace(-0.25, 1.15, 300))
    E = np.stack([A - ws[0], B - ws[1]])
    Lg = 0.5 * np.einsum("i...,ij,j...->...", E, Hs, E)
    fig, a = plt.subplots(figsize=(3.4, 3.3))
    a.contour(A, B, Lg, levels=[0.005, 0.02, 0.06, 0.12, 0.2, 0.3],
              colors=GREY_RULE, linewidths=0.6)
    P = np.array([np.zeros(2)] + [w[1:] for _, w, _ in gd])
    a.plot(P[:, 0], P[:, 1], "-o", color=GREY, alpha=0.55, lw=0.7, ms=2.5,
           label="GD = cycle-1 fast weights")
    F = np.array([rows[4][1][1:]] + [w[1:] for _, w, _ in rows[5:8]])
    a.plot(F[:, 0], F[:, 1], "--o", color=BLUE, lw=0.8, ms=2.5,
           label="cycle-2 fast weights")
    S = np.array([np.zeros(2), rows[4][1][1:], rows[8][1][1:]])
    a.plot(S[:, 0], S[:, 1], "-s", color=ORANGE, lw=1.1, ms=3.5,
           label="slow weights")
    a.plot(*ws, "*", color=RED, ms=7, label=r"$\boldsymbol{w}^\star$")
    a.set_xlabel("$w_1$")
    a.set_ylabel("$w_2$")
    a.set_xlim(-0.15, 1.35)
    a.set_ylim(-0.25, 1.15)
    a.set_aspect("equal")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7.5,
             ncol=2, columnspacing=1.0, handlelength=1.8)
    fig.tight_layout()
    save(fig, "ch19-lookahead")


def ch19_net():
    """Five methods on the 2-2-1 net: full batch eta = 1, batches of 2."""
    from ch19_stability import run, METHODS, LIN
    st = {"Adam": (GREY, "-"), "Adam + warm-up": (BLUE, "--"),
          "RAdam": (ORANGE, "-"), "Lookahead(Adam)": (GREEN, "-"),
          "AdaBelief": (RED, "--")}
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.4), sharey=True)
    for a, (b, eta, title) in zip(axs, ((None, 1.0, r"full batch, $\eta = 1$"),
                                        (2, 0.1, r"batches of 2, $\eta = 0.1$"))):
        for name in METHODS:
            h = np.log(np.clip(run(name, eta, 2000, b), 1e-12, 1e2))
            h = np.exp(np.convolve(h, np.ones(25) / 25, mode="valid"))
            col, ls = st[name]
            a.semilogy(np.arange(len(h)) + 12, h, color=col,
                       ls=ls, lw=0.8, label=name)
        a.axhline(LIN, color=GREY, lw=0.7, ls=":", label=r"linear $L^\star$")
        a.set_title(title)
        a.set_xlabel("step $t$")
        a.set_xlim(0, 2000)
        a.set_xticks([0, 1000, 2000])
        a.set_ylim(1e-6, 5)
        a.grid(lw=0.3)
    axs[0].set_ylabel("$L$, standard units")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               fontsize=7.5, ncol=3, columnspacing=1.0, handlelength=1.8)
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch19-net")


def ch20_valley():
    """GD and SAM on a valley of minima whose curvature falls to the right."""
    from ch20_sam import valley, a_val, L_val
    U, V = np.meshgrid(np.linspace(-2.6, 2.6, 400),
                       np.linspace(-1.6, 1.6, 300))
    Lg = L_val(np.stack([U, V]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.3),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    a1.contour(U, V, Lg, levels=[0.02, 0.1, 0.3, 0.7, 1.4, 2.4, 4.0],
               colors=GREY_RULE, linewidths=0.6)
    a1.axhline(0, color=GREY, lw=0.5, ls=":")
    for w0, mk in (((-2.0, 1.5), "o"), ((-1.0, -1.2), "s")):
        for r, col, ls, lab in ((0.0, GREY, "-", "GD"),
                                (0.2, BLUE, "--", "SAM, $r = 0.2$")):
            P = valley(w0, r)
            Q = P if r == 0 else P[::2]          # every second SAM step
            a1.plot(Q[:, 0], Q[:, 1], ls, color=col, lw=0.9,
                    label=lab if mk == "o" else None)
            a1.plot(*P[-1], mk, color=col, ms=3.2)
            a2.plot(np.arange(len(P)), a_val(P[:, 0]), ls, color=col, lw=0.9)
        a1.plot(*w0, mk, color=ORANGE, ms=3.2)
    a1.set_xlabel("$w_1$ (along the valley)")
    a1.set_ylabel("$w_2$")
    a1.set_xlim(-2.6, 2.6)
    a1.set_ylim(-1.6, 1.6)
    a2.set_xlabel("step $t$")
    a2.set_ylabel(r"sharpness $a(w_1)$")
    a2.set_xlim(0, 1000)
    a2.set_ylim(0, 9.5)
    a2.set_xticks([0, 500, 1000])
    a2.grid(lw=0.3)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               fontsize=7.5, ncol=2, columnspacing=1.5, handlelength=1.8)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch20-valley")


def ch20_net():
    """GD, SAM, Adam and Adam-SAM on the students, against gradient work."""
    from ch20_sam import RUNS, net_run
    from ch10_perweight import LIN
    st = {"GD": (GREY, "-"), "SAM": (BLUE, "--"), "Adam": (ORANGE, "-"),
          "Adam-SAM": (RED, "--")}
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.4), sharey=True)
    for a, (b, title) in zip(axs, ((None, "full batch"),
                                   (2, "batches of 2"))):
        for name, eta, r, kind in RUNS:
            h = net_run(eta, r, kind, b)[1]
            ev = np.arange(len(h)) * (2 if r > 0 else 1)
            col, ls = st[name]
            a.semilogy(ev, np.clip(h, 1e-12, None), color=col, ls=ls,
                       lw=0.8, label=name)
        a.axhline(LIN, color=GREY, lw=0.7, ls=":", label=r"linear $L^\star$")
        a.set_title(title)
        a.set_xlabel("gradient evaluations")
        a.set_xlim(0, 4000)
        a.set_xticks([0, 2000, 4000])
        a.set_ylim(1e-8, 2)
        a.grid(lw=0.3)
    axs[0].set_ylabel("$L$, standard units")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               fontsize=7.5, ncol=5, columnspacing=1.0, handlelength=1.8)
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch20-net")


def ch21_ns():
    """Newton-Schulz maps on one singular value: one step and five steps."""
    from ch21_matrix import ns_scalar, NS5
    s = np.linspace(0, 1.25, 400)
    cub = (1.5, -0.5, 0.0)
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.3))
    a = axs[0]
    a.plot(s, ns_scalar(s, 1, cub)[1], color=BLUE, lw=1.0, label="cubic")
    a.plot(s, ns_scalar(s, 1, NS5)[1], color=ORANGE, lw=1.0, label="quintic")
    a.plot(s, s, color=GREY, lw=0.6, ls=":", label=r"$\sigma$")
    a.axhline(1, color=GREY_RULE, lw=0.6)
    a.set_xlim(0, 1.25)
    a.set_ylim(0, 1.6)
    a.set_xlabel(r"$\sigma$")
    a.set_ylabel(r"after one step")
    a.set_title("one step")
    a.legend(loc="lower right", fontsize=7.5, handlelength=1.6)
    a = axs[1]
    s0 = np.logspace(-3, 0, 400)
    a.semilogx(s0, ns_scalar(s0, 5, cub)[-1], color=BLUE, lw=1.0)
    a.semilogx(s0, ns_scalar(s0, 5, NS5)[-1], color=ORANGE, lw=1.0)
    a.axhspan(0.7, 1.3, color=GREY_RULE, alpha=0.35, lw=0)
    a.axhline(1, color=GREY_RULE, lw=0.6)
    a.set_xlim(1e-3, 1)
    a.set_ylim(0, 1.6)
    a.set_xlabel(r"starting $\sigma$")
    a.set_ylabel("after five steps")
    a.set_title("five steps")
    for a in axs:
        a.grid(lw=0.3)
    fig.tight_layout(w_pad=1.2)
    save(fig, "ch21-ns")


def ch21_net():
    """Loss of Adam, Shampoo, SOAP and Muon on the students."""
    from ch21_matrix import run, BEST, ETA2
    from ch10_perweight import LIN
    st = {"adam": ("Adam", GREY, "-"), "shampoo": ("Shampoo", BLUE, "-"),
          "soap": ("SOAP", GREEN, "--"), "muon": ("Muon", ORANGE, "-.")}
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.4), sharey=True)
    for a, (b, title) in zip(axs, ((None, "full batch"),
                                   (2, "batches of 2"))):
        for kind, eta in BEST.items():
            h = run(kind, eta, b=b, eta2=ETA2)
            name, col, ls = st[kind]
            a.semilogy(np.clip(h, 1e-12, None), color=col, ls=ls,
                       lw=1.0 if kind == "shampoo" else 0.6, label=name,
                       zorder=4 if kind == "shampoo" else 2)
        a.axhline(LIN, color=GREY, lw=0.7, ls=":", label=r"linear $L^\star$")
        a.set_title(title)
        a.set_xlabel("step")
        a.set_xlim(0, 2000)
        a.set_xticks([0, 1000, 2000])
        a.set_ylim(1e-8, 2)
        a.grid(lw=0.3)
    axs[0].set_ylabel("$L$, standard units")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               fontsize=7.5, ncol=5, columnspacing=1.0, handlelength=1.8)
    fig.tight_layout(w_pad=1.6)
    save(fig, "ch21-net")


def ch22_hutch():
    """Running means of Hutchinson estimates of diag(H) for the students."""
    from ch22_light import hutch_path
    from students import H
    p = hutch_path()
    n = len(p["signs"])
    fig, axs = plt.subplots(1, 3, figsize=(4.8, 1.9))
    for j, a in enumerate(axs):
        k = np.arange(1, n + 1)
        a.plot(k, p["gauss"][:, j], color=ORANGE, lw=0.7, label="Gaussian")
        a.plot(k, p["signs"][:, j], color=BLUE, lw=0.9, label="signs")
        a.axhline(H[j, j], color=GREY, lw=0.7, ls=":", label="true")
        a.set_title(f"$h_{{{j + 1}{j + 1}}} = {H[j, j]:g}$")
        a.set_xlim(0, n)
        a.set_xticks([0, 100, 200])
        a.set_xlabel("probes")
        a.grid(lw=0.3)
    axs[0].set_ylim(-6, 8)
    axs[1].set_ylim(0, 25)
    axs[2].set_ylim(25, 65)
    axs[0].set_ylabel("running mean")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               fontsize=7.5, ncol=3, handlelength=1.8)
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch22-hutch")


def ch22_clip():
    """Sophia step of one coordinate against its curvature estimate."""
    s, m = 0.01, 0.02
    h = np.linspace(-6, 10, 1601)
    step = np.clip(m / np.maximum(s * h, 1e-12), -1, 1)
    fig, a = plt.subplots(figsize=(3.6, 2.1))
    hp = np.linspace(0.6, 10, 400)
    a.plot(hp, m / (s * hp), color=GREY, lw=0.7, ls="--",
           label=r"Newton-like $m/(s\,h)$")
    a.plot(h, step, color=BLUE, lw=1.2, label="Sophia step")
    a.axvline(m / s, color=GREY_RULE, lw=0.6)
    a.text(m / s + 0.2, 0.08, r"$h = m/s$", fontsize=7.5)
    a.text(-5.6, 0.62, "sign step\n(also for $h \\leq 0$)", fontsize=7.5)
    a.text(5.2, 0.62, "curvature\nshortens it", fontsize=7.5)
    a.set_xlim(-6, 10)
    a.set_ylim(0, 2)
    a.set_xlabel("curvature estimate $h$")
    a.set_ylabel(r"step $/\ \eta$")
    a.grid(lw=0.3)
    a.legend(loc="upper right", fontsize=7.5, handlelength=1.8)
    fig.tight_layout()
    save(fig, "ch22-clip")


def ch22_net():
    """Loss of AdamW, Adafactor, Lion and Sophia on the students."""
    from ch22_light import run, BEST, RHO
    from ch10_perweight import LIN
    st = {"adamw": ("AdamW", GREY, "-"), "adafactor": ("Adafactor", GREEN, "--"),
          "lion": ("Lion", ORANGE, "-."), "sophia": ("Sophia", BLUE, "-")}
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.4), sharey=True)
    for a, (b, title) in zip(axs, ((None, "full batch"),
                                   (2, "batches of 2"))):
        for kind, eta in BEST.items():
            h = run(kind, eta, RHO[kind], b=b)
            name, col, ls = st[kind]
            a.semilogy(np.clip(h, 1e-12, None), color=col, ls=ls,
                       lw=1.0 if kind == "sophia" else 0.6, label=name,
                       zorder=4 if kind == "sophia" else 2)
        a.axhline(LIN, color=GREY, lw=0.7, ls=":", label=r"linear $L^\star$")
        a.set_title(title)
        a.set_xlabel("step")
        a.set_xlim(0, 2000)
        a.set_xticks([0, 1000, 2000])
        a.set_ylim(1e-8, 2)
        a.grid(lw=0.3)
    axs[0].set_ylabel("$L$, standard units")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               fontsize=7.5, ncol=5, columnspacing=1.0, handlelength=1.8)
    fig.tight_layout(w_pad=1.6)
    save(fig, "ch22-net")


def ch23_paths():
    """The three sequences of schedule-free SGD on a noiseless quadratic."""
    from ch23_schedulefree import noiseless_paths
    r = noiseless_paths(20)
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    ax.axhline(0, color=GREY_RULE, lw=0.6)
    ax.plot(r[:, 0], r[:, 1], color=ORANGE, ls="--", lw=0.9, marker="o",
            ms=2.2, label=r"$w_t$ (SGD steps)")
    ax.plot(r[:, 0], r[:, 3], color=GREEN, ls="-.", lw=0.9, marker="s",
            ms=2.0, label=r"$u_t$ (gradient point)")
    ax.plot(r[:, 0], r[:, 2], color=BLUE, lw=1.2, marker="o", ms=2.2,
            label=r"$\bar w_t$ (average)")
    ax.set_xlabel("step $t$")
    ax.set_ylabel("weight")
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.grid(lw=0.3)
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    save(fig, "ch23-paths")


def ch23_quad():
    """Expected gaps on the noisy 1-D quadratic."""
    from ch23_schedulefree import quad, QUAD
    st = {"constant": (GREY, "-", 0.7), "linear decay": (GREEN, "--", 0.8),
          "cosine": (GREEN, "-", 0.8), "Polyak": (ORANGE, "-.", 0.9),
          "schedule-free": (BLUE, "-", 1.3), "primal": (RED, ":", 1.0)}
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    t = np.arange(1001)
    for name, (kind, kw) in QUAD.items():
        col, ls, lw = st[name]
        ax.loglog(t[1:], quad(kind, **kw)[1:], color=col, ls=ls, lw=lw,
                  label=name)
    ax.loglog(t[10:], 1 / (2 * t[10:]), color=GREY, lw=0.6, ls=(0, (1, 2)),
              label=r"$\sigma^2/(2ht)$")
    ax.set_xlabel("step $t$")
    ax.set_ylabel("expected gap")
    ax.set_xlim(1, 1000)
    ax.set_ylim(2e-4, 60)
    ax.grid(lw=0.3, which="major")
    fig.legend(loc="upper center", bbox_to_anchor=(0.55, 0.02), fontsize=7,
               ncol=4, columnspacing=1.0, handlelength=2.2)
    fig.tight_layout()
    save(fig, "ch23-quad")


def ch23_linear():
    """Linear model, batches of two: schedule-free against cosine for
    three planned lengths."""
    from ch23_schedulefree import lin_run, mean_runs, BETA
    from ch15_schedules import constant, cosine
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    t = np.arange(801)
    g = mean_runs(lambda s: lin_run("sgd", 800, sched=constant(0.04),
                                    seed=s))
    ax.semilogy(t, g, color=GREY, lw=0.7, label=r"constant $0.04$")
    g = mean_runs(lambda s: lin_run("sf", 800, eta=0.4, beta=0.0, seed=s))
    ax.semilogy(t, g, color=ORANGE, ls="-.", lw=0.9,
                label=r"Polyak, $\eta = 0.4$")
    for i, T in enumerate((200, 400, 800)):
        g = mean_runs(lambda s: lin_run("sgd", T, sched=cosine(0.4, T),
                                        seed=s))
        ax.semilogy(t[:T + 1], g, color=GREEN, ls="--", lw=0.8,
                    label="cosine, $T = 200, 400, 800$" if i == 0 else None)
        ax.plot(T, g[-1], "o", color=GREEN, ms=3)
    g = mean_runs(lambda s: lin_run("sf", 800, eta=1.0, beta=BETA, seed=s))
    ax.semilogy(t, g, color=BLUE, lw=1.3, label=r"schedule-free, $\eta = 1$")
    ax.set_xlabel("step $t$")
    ax.set_ylabel(r"mean gap, points$^2$")
    ax.set_xlim(0, 815)
    ax.set_xticks([0, 200, 400, 600, 800])
    ax.set_ylim(1e-3, 3e3)
    ax.grid(lw=0.3)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    save(fig, "ch23-linear")


def ch23_net():
    """The network, batches of two: AdamW with cosine against
    schedule-free AdamW."""
    from ch23_schedulefree import adamw, sf_adamw
    from ch15_schedules import cosine
    from ch10_perweight import LIN
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    log = []
    h = sf_adamw(0.1, 2000, log=log)
    ax.semilogy(np.r_[np.nan, [r[1] for r in log]], color=ORANGE, lw=0.4,
                alpha=0.55, label=r"schedule-free, $L(\boldsymbol{w}_t)$")
    for T, col, ls in ((2000, GREEN, "-"), (500, GREY, "--")):
        g = adamw(cosine(0.1, T, tw=100), T)
        ax.semilogy(g, color=col, ls=ls, lw=0.9,
                    label=f"AdamW + cosine, $T = {T}$")
    ax.semilogy(h, color=BLUE, lw=1.3, label=r"schedule-free, $L(\bar{\boldsymbol{w}}_t)$")
    ax.axhline(LIN, color=GREY, lw=0.7, ls=":", label=r"linear $L^\star$")
    ax.set_xlabel("step $t$")
    ax.set_ylabel("$L$, standard units")
    ax.set_xlim(0, 2000)
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_ylim(3e-5, 2)
    ax.grid(lw=0.3)
    fig.legend(loc="upper center", bbox_to_anchor=(0.55, 0.02), fontsize=7,
               ncol=2, columnspacing=1.2, handlelength=2.2)
    fig.tight_layout()
    save(fig, "ch23-net")


ERA24 = {2: "Classical era", 3: "Neural-network era", 4: "Adaptive era",
         5: "Large-scale and modern era"}
COL24 = [BLUE, ORANGE, GREEN, RED, GREY, "#6A3D9A", "#2A8FA0"]
LS24 = ["-", "--", "-", "-.", "-", "--", ":"]


def ch24_curves():
    """Mean training loss of the final runs, one panel per era."""
    import ch24_case as c
    res = c.results()
    fig, axs = plt.subplots(4, 1, figsize=(4.8, 6.0), sharex=True)
    for ax, era in zip(axs, (2, 3, 4, 5)):
        kinds = [k for k, v in c.METHODS.items() if v[2] == era]
        for i, k in enumerate(kinds):
            h = np.mean([r[0] for r in res["final"][k]], axis=0)
            lab = c.METHODS[k][0].replace("$-$", "-")
            ax.semilogy(np.maximum(h, 1e-4), color=COL24[i], ls=LS24[i],
                        lw=1.0, label=lab)
        ax.axhline(c.TARGET, color=GREY, lw=0.6, ls=":")
        ax.set_title(ERA24[era], loc="left", fontsize=8.5)
        ax.set_ylim(6e-5, 2)
        ax.set_yticks([1e-4, 1e-2, 1])
        ax.set_ylabel("training loss")
        ax.grid(lw=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7,
                  handlelength=2.2, labelspacing=0.3)
    axs[-1].set_xlabel("epochs of gradient evaluations")
    axs[-1].set_xlim(0, c.EPOCHS)
    fig.tight_layout(h_pad=0.6)
    save(fig, "ch24-curves")


def ch24_methods():
    """Test loss and epochs to the target loss, mean and spread over seeds."""
    import ch24_case as c
    res = c.results()
    kinds = list(c.METHODS)
    ys = np.arange(len(kinds))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 3.6), sharey=True)
    for y, k in zip(ys, kinds):
        tr, te, ac, hits = c.summary(res, k)
        col = COL24[c.METHODS[k][2] - 2]
        if te.mean() < 1e-4:
            a1.annotate("", xy=(1.1e-4, y), xytext=(2.4e-4, y),
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
        else:
            a1.errorbar(te.mean(), y, xerr=te.std(), fmt="o", ms=3,
                        color=col, elinewidth=0.8, capsize=1.5)
        h = np.array(hits, float)
        a2.errorbar(np.median(h), y, xerr=[[np.median(h) - h.min()],
                    [h.max() - np.median(h)]], fmt="o", ms=3, color=col,
                    elinewidth=0.8, capsize=1.5)
    a1.set_xscale("log")
    a1.set_xlim(1e-4, 0.1)
    a1.set_xlabel("test loss")
    a2.set_xlim(0, 200)
    a2.set_xlabel("epochs to training loss 0.05")
    a1.set_yticks(ys)
    a1.set_yticklabels([c.METHODS[k][0].replace("$-$", "-") for k in kinds],
                       fontsize=7.5)
    for a in (a1, a2):
        a.grid(lw=0.3, axis="x")
        a.set_ylim(-0.7, len(kinds) - 0.3)
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch24-methods")


def ch24_eta():
    """Mean validation loss after the budget against the learning rate."""
    import ch24_case as c
    res = c.results()
    pick = ("gd", "minibatch", "adam", "muon", "lion", "sfadamw")
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for i, k in enumerate(pick):
        v = [np.mean(res["tune"][k][str(e)]) for e in c.GRID]
        v = np.where(np.isfinite(v), v, np.nan)
        ax.loglog(c.GRID, v, color=COL24[i], ls=LS24[i], lw=1.0, marker="o",
                  ms=2.5, label=c.METHODS[k][0])
    ax.set_xlabel(r"learning rate $\eta$")
    ax.set_ylabel("validation loss")
    ax.set_ylim(0.08, 5)
    ax.set_xlim(7e-5, 4.5)
    ax.grid(lw=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7,
              handlelength=2.2)
    fig.tight_layout()
    save(fig, "ch24-eta")


if __name__ == "__main__":
    pick = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, fn in list(globals().items()):
        if re.match(r"ch\d\d_", name) and name.startswith(pick):
            fn()
