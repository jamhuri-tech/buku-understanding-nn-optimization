"""Chapter 24: case study.  Nineteen optimisers of Chapters 3-23 train the
13-8-3 softmax network on the UCI Wine data under one protocol.

The sweep and the final runs take a few minutes; their results are cached
in cache/ch24.json (delete it to recompute).  Wall-clock times differ
between machines and runs; all other numbers are reproducible.
"""
import json
import os
import time
import numpy as np
from winenet import (X_ALL, Y_ALL, X_TR, Y_TR, X_TE, Y_TE, X_FIT, X_VAL,
                     NAMES, TUNE, FINAL, init, loss, accuracy, grads, batches,
                     forward, softmax, tilde, onehot)
from optimizers import update
from ch14_adam import OM1, OM2, ZETA

np.set_printoptions(precision=4, suppress=True)
EPOCHS = 200                       # budget in passes of gradient evaluations
SEEDS = range(5)                   # initialisation and batch-order seeds
GRID = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
B = 16                             # mini-batch size
TARGET = 0.05                      # target training loss
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "cache", "ch24.json")

# key: (label, chapter, era, batch size, gradient evaluations per step)
METHODS = {
    "gd":        ("GD", 3, 2, None, 1),
    "sgd":       ("SGD", 5, 2, 1, 1),
    "heavy":     ("heavy ball", 6, 2, B, 1),
    "nesterov":  ("Nesterov", 8, 3, B, 1),
    "rprop":     ("iRprop$-$", 10, 3, None, 1),
    "minibatch": ("mini-batch SGD", 11, 3, B, 1),
    "adagrad":   ("AdaGrad", 12, 4, B, 1),
    "rmsprop":   ("RMSProp", 13, 4, B, 1),
    "adam":      ("Adam", 14, 4, B, 1),
    "adamcos":   ("Adam + cosine", 15, 4, B, 1),
    "amsgrad":   ("AMSGrad", 16, 4, B, 1),
    "adamw":     ("AdamW", 17, 4, B, 1),
    "lamb":      ("LAMB", 18, 5, B, 1),
    "radam":     ("RAdam", 19, 5, B, 1),
    "sam":       ("SAM + Adam", 20, 5, B, 2),
    "muon":      ("Muon + Adam", 21, 5, B, 1),
    "lion":      ("Lion", 22, 5, B, 1),
    "adafactor": ("Adafactor", 22, 5, B, 1),
    "sfadamw":   ("schedule-free AdamW", 23, 5, B, 1),
}


def train(kind, eta, seed, data, epochs=EPOCHS, eta2=None, curves=False):
    """Returns the final weights and, if curves, the training loss, test
    loss and test accuracy after every pass (index = passes of gradient
    evaluations, so SAM records every second index)."""
    (X, T), (Xv, Tv) = data
    _, _, _, b, ev = METHODS[kind]
    W, st = init(seed), [{}, {}]
    rng = np.random.default_rng(1000 + seed)
    passes = epochs // ev
    per = len(batches(len(T), b, np.random.default_rng(0)))
    hist = [[loss(W, X, T)], [loss(W, Xv, Tv)], [accuracy(W, Xv, Tv)]]
    t = 0
    for _ in range(passes):
        for idx in batches(len(T), b, rng):
            t += 1
            W = update(kind, W, lambda V: grads(V, X[idx], T[idx]), st, t,
                       eta, passes * per, eta2)
        lt = loss(W, X, T)
        if not np.isfinite(lt) or lt > 1e3:
            return None, None
        if curves:
            for _ in range(ev):
                hist[0].append(lt)
                hist[1].append(loss(W, Xv, Tv))
                hist[2].append(accuracy(W, Xv, Tv))
    return W, hist


def run_all():
    """Sweep on (fit, validation), then final runs on (train, test)."""
    res = {"tune": {}, "eta": {}, "final": {}, "time": {}}
    for kind in METHODS:
        e2 = res["eta"].get("adam")
        row = {}
        for eta in GRID:
            vals = []
            for s in SEEDS:
                W, _ = train(kind, eta, s, TUNE, eta2=e2)
                vals.append(np.inf if W is None else loss(W, *TUNE[1]))
            row[str(eta)] = vals
        res["tune"][kind] = row
        best = min(GRID, key=lambda e: np.mean(row[str(e)]))
        res["eta"][kind] = best
        runs, secs = [], []
        for s in SEEDS:
            t0 = time.perf_counter()
            _, h = train(kind, best, s, FINAL, eta2=e2, curves=True)
            secs.append(time.perf_counter() - t0)
            runs.append(h)
        res["final"][kind] = runs
        res["time"][kind] = float(np.median(secs))
        print(kind, best, flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(res, f, default=lambda x: None if not np.isfinite(x) else x)
    return res


def results():
    if not os.path.exists(CACHE):
        return run_all()
    with open(CACHE) as f:
        res = json.load(f)
    for row in res["tune"].values():
        for k, v in row.items():
            row[k] = [np.inf if x is None else x for x in v]
    return res


def to_target(h, target=TARGET):
    """First pass at which the training loss is at most target."""
    hit = np.nonzero(np.array(h) <= target)[0]
    return int(hit[0]) if len(hit) else None


def summary(res, kind):
    runs = res["final"][kind]
    tr = np.array([r[0][-1] for r in runs])
    te = np.array([r[1][-1] for r in runs])
    ac = np.array([r[2][-1] for r in runs])
    hits = [to_target(r[0]) for r in runs]
    return tr, te, ac, hits


# ------------------------------ printing ---------------------------------
def show_data():
    print("samples, features:", X_ALL.shape)
    print("per class, all/train/test:", np.bincount(Y_ALL),
          np.bincount(Y_TR), np.bincount(Y_TE))
    print("fit/validation:", len(X_FIT), len(X_VAL))
    cols = (0, 1, 6, 9, 12)
    print("rows (1-based) and columns", [NAMES[c] for c in cols])
    for c in range(3):
        for i in np.nonzero(Y_ALL == c)[0][:2]:
            print(f"  {i + 1:4d}", X_ALL[i, list(cols)], c)
    mu, sd = X_TR.mean(axis=0), X_TR.std(axis=0)
    print(f"{'feature':30s}{'min':>9s}{'max':>9s}{'mean':>9s}{'sd':>9s}")
    for j, nm in enumerate(NAMES):
        print(f"{nm:30s}{X_ALL[:, j].min():9.2f}{X_ALL[:, j].max():9.2f}"
              f"{mu[j]:9.3f}{sd[j]:9.3f}")


HAND_COLS = (0, 6)                       # alcohol, flavanoids
HW1 = np.array([[0.1, 0.5, -0.5],
                [-0.1, 0.3, 0.4]])
HW2 = np.array([[0.0, 0.6, -0.4],
                [0.1, -0.5, -0.3],
                [-0.1, 0.2, 0.7]])


def hand(eta=0.01):
    """Example 24.x: one sample, a 2-2-3 network, one Adam step."""
    i = 0
    raw = X_TR[i, list(HAND_COLS)]
    mu = X_TR[:, list(HAND_COLS)].mean(axis=0)
    sd = X_TR[:, list(HAND_COLS)].std(axis=0)
    x = np.round((raw - mu) / sd, 2)
    print("sample: raw", raw, "class", Y_TR[i])
    print("mean", np.round(mu, 3), "sd", np.round(sd, 3), "x", x)
    xt = np.concatenate([[1.0], x])
    z = HW1 @ xt
    a = np.tanh(z)
    at = np.concatenate([[1.0], a])
    s = HW2 @ at
    p = softmax(s[None])[0]
    t = onehot(np.array([Y_TR[i]]))[0]
    print("z", z, "a", a)
    print("s", s, "p", p, "loss", -np.log(p[Y_TR[i]]))
    e = p - t
    G2 = np.outer(e, at)
    d = (HW2[:, 1:].T @ e) * (1 - a * a)
    G1 = np.outer(d, xt)
    print("e", e)
    print("G2\n", G2)
    print("Wbar2^T e", HW2[:, 1:].T @ e, "1 - a*a", 1 - a * a, "delta", d)
    print("G1\n", G1)
    W, X, T = [HW1.copy(), HW2.copy()], xt[None], t[None]
    G = grads(W, X, T)
    print("check", max(np.abs(G[0] - G1).max(), np.abs(G[1] - G2).max()))
    g = G1[0, 1]
    m = (1 - OM1) * g
    r = (1 - OM2) * g * g
    mh, rh = m / (1 - OM1), r / (1 - OM2)
    print(f"Adam on w(1)_11: g={g:.6f} m={m:.7f} r={r:.4e} "
          f"mhat={mh:.6f} sqrt(rhat)={np.sqrt(rh):.6f} "
          f"step={eta * mh / (np.sqrt(rh) + ZETA):.6f} "
          f"new={HW1[0, 1] - eta * mh / (np.sqrt(rh) + ZETA):.4f} "
          f"GD step={eta * g:.6f}")


def show_tune(res):
    print("mean validation loss after the budget (inf = diverged)")
    print(f"{'method':14s}" + "".join(f"{e:>8g}" for e in GRID))
    for kind, row in res["tune"].items():
        cells = []
        for e in GRID:
            v = np.mean(row[str(e)])
            cells.append(f"{v:8.3f}" if np.isfinite(v) else f"{'div':>8s}")
        print(f"{kind:14s}" + "".join(cells) + f"  -> {res['eta'][kind]:g}")


def window(res, kind, tol=1.2):
    """Grid rates whose mean validation loss is within tol x the best."""
    row = res["tune"][kind]
    v = np.array([np.mean(row[str(e)]) for e in GRID])
    ok = [e for e, x in zip(GRID, v) if x <= tol * v.min()]
    return v.min(), ok


def show_window(res):
    print("best mean validation loss, rates within 1.2 x best")
    for kind in METHODS:
        best, ok = window(res, kind)
        print(f"{kind:12s}{best:8.3f}  {len(ok)}  {min(ok):g} .. {max(ok):g}")


def show_final(res):
    print(f"{'method':12s}{'eta':>7s}{'train':>16s}{'test loss':>16s}"
          f"{'test acc %':>12s}{'to L*':>7s}{'hit':>4s}{'sec':>6s}")
    for kind in METHODS:
        tr, te, ac, hits = summary(res, kind)
        ok = [h for h in hits if h is not None]
        med = f"{int(np.median(ok)):7d}" if ok else f"{'-':>7s}"
        print(f"{kind:12s}{res['eta'][kind]:7g}"
              f"{tr.mean():9.4f}+-{tr.std():.4f}"
              f"{te.mean():9.4f}+-{te.std():.4f}"
              f"{100 * ac.mean():7.1f}+-{100 * ac.std():.1f}"
              f"{med}{len(ok):4d}{res['time'][kind]:6.2f}")


def show_mistakes(res):
    """Which test samples are misclassified, over all methods and seeds."""
    (X, T), (Xt, Tt) = FINAL
    tot = 0
    for kind in METHODS:
        for s in SEEDS:
            W, _ = train(kind, res["eta"][kind], s, FINAL,
                         eta2=res["eta"]["adam"])
            tot += np.sum(forward(W, Xt)[2].argmax(1) != Tt.argmax(1))
    print("misclassified test samples over 95 runs:", tot,
          "of", 95 * len(Tt))


if __name__ == "__main__":
    show_data()
    hand()
    res = results()
    show_tune(res)
    show_window(res)
    show_final(res)
