"""The 2-2-1 tanh network on the eight students (Chapter 9 onwards).

Features and target are standardised (population standard deviation).
Weights: W1 is h x (n+1) with the bias in column 0, w2 has h+1 entries
with the bias first.  Losses are in standard units; multiply by
SY**2 for squared points.
"""
import numpy as np
from students import x1, x2, y, m

MX = np.array([x1.mean(), x2.mean()])
SX = np.array([x1.std(), x2.std()])
YM, SY = y.mean(), y.std()
Z = (np.column_stack([x1, x2]) - MX) / SX          # standardised features
Xs = np.column_stack([np.ones(m), Z])              # X tilde, m x 3
ys = (y - YM) / SY                                 # standardised target

SEED = 9


def init(seed=SEED, scale=0.5, h=2):
    """Small random weights, rounded to two decimals."""
    rng = np.random.default_rng(seed)
    W1 = np.round(scale * rng.standard_normal((h, 3)), 2)
    w2 = np.round(scale * rng.standard_normal(h + 1), 2)
    return W1, w2


def forward(W1, w2, X=Xs):
    """Returns pre-activations Z (m x h), A tilde (m x (h+1)), outputs."""
    Zh = X @ W1.T
    At = np.column_stack([np.ones(len(X)), np.tanh(Zh)])
    return Zh, At, At @ w2


def loss(W1, w2, X=Xs, t=ys):
    r = forward(W1, w2, X)[2] - t
    return r @ r / (2 * len(t))


def backward(W1, w2, X=Xs, t=ys):
    """Batch gradients (G1 same shape as W1, g2 same shape as w2)."""
    _, At, yh = forward(W1, w2, X)
    d2 = yh - t                                    # m output errors
    g2 = At.T @ d2 / len(t)
    D1 = np.outer(d2, w2[1:]) * (1 - At[:, 1:] ** 2)   # m x h
    G1 = D1.T @ X / len(t)
    return G1, g2


def fd_grad(W1, w2, eps=1e-5):
    """Central differences, one parameter at a time."""
    G1, g2 = np.zeros_like(W1), np.zeros_like(w2)
    for P, G in ((W1, G1), (w2, g2)):
        for k in np.ndindex(P.shape):
            old = P[k]
            P[k] = old + eps
            up = loss(W1, w2)
            P[k] = old - eps
            dn = loss(W1, w2)
            P[k] = old
            G[k] = (up - dn) / (2 * eps)
    return G1, g2


def train(eta, steps, gam=0.0, nesterov=False, W1=None, w2=None):
    """Full-batch GD (gam=0), heavy ball, or Nesterov in velocity form.

    Returns the final weights and the list of losses L(w_t), t=0..steps.
    """
    if W1 is None:
        W1, w2 = init()
    W1, w2 = W1.copy(), w2.copy()
    V1, v2 = np.zeros_like(W1), np.zeros_like(w2)
    hist = [loss(W1, w2)]
    for _ in range(steps):
        if nesterov:
            G1, g2 = backward(W1 + gam * V1, w2 + gam * v2)
        else:
            G1, g2 = backward(W1, w2)
        V1 = gam * V1 - eta * G1
        v2 = gam * v2 - eta * g2
        W1, w2 = W1 + V1, w2 + v2
        hist.append(loss(W1, w2))
    return W1, w2, hist


def flat(W1, w2):
    """All 9 weights in one column: rows of W1, then w2 (Chapter 10)."""
    return np.r_[W1.ravel(), w2]


def unflat(p, h=2):
    return p[:3 * h].reshape(h, 3), p[3 * h:]


def loss_flat(p):
    return loss(*unflat(p))


def grad_flat(p, idx=None):
    """Batch gradient as one vector; idx picks a mini-batch of students."""
    X = Xs if idx is None else Xs[idx]
    t = ys if idx is None else ys[idx]
    return flat(*backward(*unflat(p), X, t))


def jacobian(p):
    """Rows are the gradients of the outputs, d yhat_i / d w (m x 9)."""
    W1, w2 = unflat(p)
    yh = forward(W1, w2)[2]
    return np.array([flat(*backward(W1, w2, Xs[[i]], yh[[i]] - 1))
                     for i in range(len(ys))])


def sample_grads(p):
    """Rows are the per-student gradients g_i = r_i * d yhat_i / d w."""
    r = forward(*unflat(p))[2] - ys
    return r[:, None] * jacobian(p)


def hess_flat(p, eps=1e-5):
    """Hessian of the loss by central differences of the gradient."""
    P = len(p)
    Hm = np.zeros((P, P))
    for j in range(P):
        e = np.zeros(P)
        e[j] = eps
        Hm[:, j] = (grad_flat(p + e) - grad_flat(p - e)) / (2 * eps)
    return (Hm + Hm.T) / 2
