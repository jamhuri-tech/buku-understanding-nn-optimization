"""The update rules of Chapters 3-23 for a list of weight matrices
(Chapter 24).  Adam, Muon, AdamW, Adafactor and Lion are imported
unchanged from Chapters 21 and 22; the constants come from Chapters 13,
14 and 19.  Every rule sees the bias in column 0 of each matrix; the
weight decay skips it."""
import numpy as np
from ch13_rmsprop import OMEGA
from ch14_adam import OM1, OM2, ZETA
from ch15_schedules import cosine
from ch19_stability import q_t, q_inf, nu_t
from ch21_matrix import adam_step, muon_step
from ch22_light import adamw_step, adafactor_step, lion_step

GAM = 0.9        # momentum of the heavy ball and of Nesterov
RHO = 0.01       # decoupled weight decay of AdamW, LAMB, schedule-free
RHO_LION = 0.1   # weight decay of Lion
RADIUS = 0.05    # SAM radius r
CHI = 0.9        # schedule-free interpolation
WARM = 0.05      # warm-up as a fraction of the run
UP, DOWN, DMAX, DMIN = 1.2, 0.5, 50.0, 1e-6      # Rprop (Chapter 10)


def mask(P):
    M = np.ones_like(P)
    M[:, 0] = 0.0
    return M


def amsgrad_step(P, g, st, t, eta):
    st["m"] = OM1 * st.get("m", 0 * P) + (1 - OM1) * g
    st["r"] = OM2 * st.get("r", 0 * P) + (1 - OM2) * g * g
    st["rm"] = np.maximum(st.get("rm", 0 * P), st["r"])
    den = np.sqrt(st["rm"] / (1 - OM2 ** t)) + ZETA
    return P - eta * (st["m"] / (1 - OM1 ** t)) / den


def radam_step(P, g, st, t, eta, thr=4.0):
    st["m"] = OM1 * st.get("m", 0 * P) + (1 - OM1) * g
    st["r"] = OM2 * st.get("r", 0 * P) + (1 - OM2) * g * g
    mh = st["m"] / (1 - OM1 ** t)
    q = q_t(t, OM2)
    if q <= thr:
        return P - eta * mh
    rh = st["r"] / (1 - OM2 ** t)
    return P - eta * nu_t(q, q_inf(OM2)) * mh / (np.sqrt(rh) + ZETA)


def lamb_step(P, g, st, t, eta):
    st["m"] = OM1 * st.get("m", 0 * P) + (1 - OM1) * g
    st["r"] = OM2 * st.get("r", 0 * P) + (1 - OM2) * g * g
    u = ((st["m"] / (1 - OM1 ** t)) / (np.sqrt(st["r"] / (1 - OM2 ** t))
                                       + ZETA) + RHO * mask(P) * P)
    nw, nu = np.linalg.norm(P), np.linalg.norm(u)
    return P - eta * (nw / nu if nw > 0 and nu > 0 else 1.0) * u


def rprop_step(P, g, st, t, eta):
    """iRprop-: eta is the initial step Delta_0."""
    D = st.get("D", np.full(P.shape, eta))
    s = st.get("g", 0 * P) * g
    D = np.where(s > 0, np.minimum(UP * D, DMAX),
                 np.where(s < 0, np.maximum(DOWN * D, DMIN), D))
    g = np.where(s < 0, 0.0, g)
    st["D"], st["g"] = D, g
    return P - np.sign(g) * D


def plain(P, g, st, t, eta):
    return P - eta * g


def adagrad_step(P, g, st, t, eta):
    st["h"] = st.get("h", 0 * P) + g * g
    return P - eta * g / (np.sqrt(st["h"]) + ZETA)


def rmsprop_step(P, g, st, t, eta):
    st["r"] = OMEGA * st.get("r", 0 * P) + (1 - OMEGA) * g * g
    return P - eta * g / (np.sqrt(st["r"]) + ZETA)


def heavy_step(P, g, st, t, eta):
    st["v"] = GAM * st.get("v", 0 * P) - eta * g
    return P + st["v"]


SIMPLE = {"gd": plain, "sgd": plain, "minibatch": plain, "heavy": heavy_step,
          "rprop": rprop_step, "adagrad": adagrad_step,
          "rmsprop": rmsprop_step, "adam": adam_step, "amsgrad": amsgrad_step,
          "radam": radam_step, "lamb": lamb_step}


def update(kind, W, grad, st, t, eta, T, eta2=None):
    """One step of method `kind` at step t of T on the weight list W.

    grad(W) returns the mini-batch gradients at W; st holds one dict per
    matrix.  Returns the new list of weights.  For the schedule-free
    method the returned list is the average w bar."""
    if kind in SIMPLE:
        return [SIMPLE[kind](P, g, s, t, eta)
                for P, g, s in zip(W, grad(W), st)]
    if kind == "nesterov":                 # look-ahead form of Chapter 8
        U = [P + GAM * (P - s.get("prev", P)) for P, s in zip(W, st)]
        for P, s in zip(W, st):
            s["prev"] = P
        return [Pu - eta * g for Pu, g in zip(U, grad(U))]
    if kind == "adamcos":
        e = cosine(eta, T, tw=max(1, round(WARM * T)))(t - 1)
        return [adam_step(P, g, s, t, e) for P, g, s in zip(W, grad(W), st)]
    if kind == "adamw":
        return [adamw_step(P, g, s, t, eta, RHO, mask(P))
                for P, g, s in zip(W, grad(W), st)]
    if kind == "lion":
        return [lion_step(P, g, s, t, eta, RHO_LION, mask(P))
                for P, g, s in zip(W, grad(W), st)]
    if kind == "adafactor":
        return [adafactor_step(P, g, s, t, eta, 0.0, mask(P))
                for P, g, s in zip(W, grad(W), st)]
    if kind == "muon":                     # Muon on W1, Adam on W2
        G = grad(W)
        return [muon_step(W[0], G[0], st[0], t, eta),
                adam_step(W[1], G[1], st[1], t, eta2)]
    if kind == "sam":                      # SAM with Adam as the base
        G = grad(W)
        nrm = np.sqrt(sum(np.sum(g * g) for g in G))
        G = grad([P + RADIUS * g / nrm for P, g in zip(W, G)])
        return [adam_step(P, g, s, t, eta) for P, g, s in zip(W, G, st)]
    if kind == "sfadamw":                  # Chapter 23, weights eta_t^2
        tw = max(1, round(WARM * T))
        e = eta * np.sqrt(1 - OM2 ** t) * min(1.0, t / tw)
        for P, s in zip(W, st):
            s.setdefault("w", P.copy())
        U = [(1 - CHI) * s["w"] + CHI * P for P, s in zip(W, st)]
        out = []
        for P, u, g, s in zip(W, U, grad(U), st):
            s["r"] = OM2 * s.get("r", 0 * P) + (1 - OM2) * g * g
            s["w"] = (s["w"] - e * g / (np.sqrt(s["r"]) + ZETA)
                      - e * RHO * mask(P) * u)
            s["S"] = s.get("S", 0.0) + e * e
            c = e * e / s["S"]
            out.append((1 - c) * P + c * s["w"])
        return out
    raise ValueError(kind)
