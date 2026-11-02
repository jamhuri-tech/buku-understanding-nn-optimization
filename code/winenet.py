"""The 13-h-3 softmax network on the UCI Wine data (Chapter 24).

Data: Aeberhard and Forina (1991), CC BY 4.0, doi:10.24432/C5PC7J, as
distributed with scikit-learn.  Rows are samples, the cultivar (0, 1, 2)
is the target.  Weights: W1 is h x (n+1) and W2 is 3 x (h+1), each with
the bias in column 0.
"""
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

SPLIT = 24                     # seed of both stratified splits
H = 8                          # hidden units
K = 3                          # classes
_d = load_wine()
X_ALL, Y_ALL = _d.data, _d.target
NAMES = _d.feature_names
N = X_ALL.shape[1]

# 124 training and 54 test samples; then 93 fit and 31 validation samples
X_TR, X_TE, Y_TR, Y_TE = train_test_split(
    X_ALL, Y_ALL, test_size=54, stratify=Y_ALL, random_state=SPLIT)
X_FIT, X_VAL, Y_FIT, Y_VAL = train_test_split(
    X_TR, Y_TR, test_size=31, stratify=Y_TR, random_state=SPLIT)


def tilde(Z):
    """Put a column of ones in front."""
    return np.column_stack([np.ones(len(Z)), Z])


def onehot(y, k=K):
    return np.eye(k)[y]


def prepare(Xa, ya, Xb, yb):
    """Standardise both sets with the mean and population standard
    deviation of the first set only.  Returns X tilde and one-hot T."""
    mu, sd = Xa.mean(axis=0), Xa.std(axis=0)
    return ((tilde((Xa - mu) / sd), onehot(ya)),
            (tilde((Xb - mu) / sd), onehot(yb)))


TUNE = prepare(X_FIT, Y_FIT, X_VAL, Y_VAL)       # (fit, validation)
FINAL = prepare(X_TR, Y_TR, X_TE, Y_TE)          # (train, test)


def init(seed, h=H, n=N, k=K):
    """Gaussian weights with standard deviation 1/sqrt(fan-in), zero biases."""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((h, n + 1)) / np.sqrt(n)
    W2 = rng.standard_normal((k, h + 1)) / np.sqrt(h)
    W1[:, 0] = 0.0
    W2[:, 0] = 0.0
    return [W1, W2]


def softmax(S):
    E = np.exp(S - S.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def forward(W, X):
    """Pre-activations Z (m x h), A tilde (m x (h+1)), probabilities P."""
    Z = X @ W[0].T
    At = tilde(np.tanh(Z))
    return Z, At, softmax(At @ W[1].T)


def loss(W, X, T):
    P = forward(W, X)[2]
    return -np.sum(T * np.log(np.maximum(P, 1e-300))) / len(T)


def accuracy(W, X, T):
    return np.mean(forward(W, X)[2].argmax(axis=1) == T.argmax(axis=1))


def grads(W, X, T):
    """Mini-batch gradients: E = P - T, G2 = E^T At / m,
    Delta = (E W2bar) * (1 - A*A), G1 = Delta^T X / m."""
    _, At, P = forward(W, X)
    E = P - T
    G2 = E.T @ At / len(T)
    D = (E @ W[1][:, 1:]) * (1 - At[:, 1:] ** 2)
    G1 = D.T @ X / len(T)
    return [G1, G2]


def fd_grads(W, X, T, eps=1e-6):
    """Central differences, one weight at a time."""
    G = [np.zeros_like(P) for P in W]
    for P, Gp in zip(W, G):
        for k in np.ndindex(P.shape):
            old = P[k]
            P[k] = old + eps
            up = loss(W, X, T)
            P[k] = old - eps
            dn = loss(W, X, T)
            P[k] = old
            Gp[k] = (up - dn) / (2 * eps)
    return G


def batches(m, b, rng):
    """Index arrays of size b (the last may be smaller); a fresh shuffle
    every epoch.  b = None gives the full batch."""
    if b is None:
        return [np.arange(m)]
    perm = rng.permutation(m)
    return [perm[k:k + b] for k in range(0, m, b)]
