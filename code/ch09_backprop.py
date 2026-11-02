"""Chapter 9: backpropagation and momentum on the 2-2-1 tanh network."""
import numpy as np
from students import m
from net221 import (Xs, ys, SX, SY, YM, init, forward, loss, backward,
                    fd_grad, train)

np.set_printoptions(precision=4, suppress=True)
LIN = 0.289198 / SY ** 2          # linear L* in standard units


def swap(W1, w2):
    """Exchange the two hidden units."""
    return W1[::-1].copy(), np.r_[w2[0], w2[2], w2[1]]


def flip(W1, w2, j=0):
    """Change the sign of hidden unit j (incoming and outgoing)."""
    W1, w2 = W1.copy(), w2.copy()
    W1[j] *= -1
    w2[j + 1] *= -1
    return W1, w2


def hessian(W1, w2, eps=1e-5):
    """9 x 9 Hessian by central differences of the gradient."""
    p = np.r_[W1.ravel(), w2]
    def g(p):
        G1, g2 = backward(p[:6].reshape(2, 3), p[6:])
        return np.r_[G1.ravel(), g2]
    H = np.zeros((9, 9))
    for k in range(9):
        e = np.zeros(9)
        e[k] = eps
        H[:, k] = (g(p + e) - g(p - e)) / (2 * eps)
    return (H + H.T) / 2


if __name__ == "__main__":
    print("std of study, sleep:", SX.round(4), " std of y:", round(SY, 4))
    print("s_y^2 =", round(SY ** 2, 4), " linear L* =", round(LIN, 6))
    print("standardised data (z1, z2, y):")
    for i in range(m):
        print(f"  {i + 1}  {Xs[i, 1]:8.4f} {Xs[i, 2]:8.4f} {ys[i]:8.4f}")

    W1, w2 = init()
    print("initial W1:\n", W1, "\ninitial w2:", w2)
    print("L(w_0) =", round(loss(W1, w2), 4))

    # Student 1 by hand
    x = Xs[0]
    z = W1 @ x
    a = np.tanh(z)
    at = np.r_[1, a]
    yh = w2 @ at
    d2 = yh - ys[0]
    print("student 1: z =", z, " a =", a, " yhat =", round(yh, 4),
          " delta2 =", round(d2, 4))
    print("  grad w2 =", d2 * at)
    d1 = w2[1:] * d2 * (1 - a ** 2)
    print("  1 - a^2 =", 1 - a ** 2, " delta1 =", d1)
    print("  grad W1 =\n", np.outer(d1, x))
    print("  loss of student 1 =", round(d2 ** 2 / 2, 4))

    # Batch gradient and finite differences
    _, At, yhat = forward(W1, w2)
    print("outputs:", yhat)
    print("errors:", yhat - ys)
    G1, g2 = backward(W1, w2)
    print("batch grad W1:\n", G1, "\nbatch grad w2:", g2)
    F1, f2 = fd_grad(W1, w2)
    diff = max(abs(F1 - G1).max(), abs(f2 - g2).max())
    print(f"largest gap backprop vs differences: {diff:.1e}")

    # One GD step, then the second step of GD and of the heavy ball
    eta, gam = 0.1, 0.9
    W1a, w2a = W1 - eta * G1, w2 - eta * g2
    print("after one step: W1 =\n", W1a, "\nw2 =", w2a)
    print("L(w_1) =", round(loss(W1a, w2a), 4))
    H1, h2 = backward(W1a, w2a)
    print("grad at w_1: W1\n", H1, "\nw2", h2)
    W1g, w2g = W1a - eta * H1, w2a - eta * h2
    W1m = W1a - eta * H1 + gam * (W1a - W1)
    w2m = w2a - eta * h2 + gam * (w2a - w2)
    print("GD w_2: W1 =\n", W1g, "\nw2 =", w2g)
    print("GD L(w_2) =", round(loss(W1g, w2g), 4))
    print("HB w_2: W1 =\n", W1m, "\nw2 =", w2m)
    print("HB L(w_2) =", round(loss(W1m, w2m), 4))

    # Training curves
    T = 2000
    runs = {"GD": train(eta, T), "HB": train(eta, T, gam),
            "NAG": train(eta, T, gam, nesterov=True)}
    for k, (V1, v2, h) in runs.items():
        h = np.array(h)
        first = [int(np.argmax(h < th)) if (h < th).any() else -1
                 for th in (LIN, 1e-3, 1e-4)]
        print(f"{k}: steps to L*lin, 1e-3, 1e-4: {first};"
              f" L(100) = {h[100]:.5f}; L({T}) = {h[-1]:.2e}"
              f" = {h[-1] * SY ** 2:.2e} points^2")
    V1, v2, _ = runs["GD"]
    print("GD final W1:\n", V1, "\nGD final w2:", v2)
    print("predictions (points):", (YM + SY * forward(V1, v2)[2]).round(2))

    # Symmetries
    S1, s2 = swap(V1, v2)
    F1, f2 = flip(V1, v2)
    print(f"loss at GD end {loss(V1, v2):.6e}, swapped {loss(S1, s2):.6e},"
          f" flipped {loss(F1, f2):.6e}")
    mid = loss((V1 + S1) / 2, (v2 + s2) / 2)
    print(f"loss at the midpoint with the swapped copy: {mid:.4f}")

    # The all-zero point
    Z1, z2 = np.zeros((2, 3)), np.zeros(3)
    G1, g2 = backward(Z1, z2)
    c = Xs.T @ ys / m
    print("all-zero: loss", round(loss(Z1, z2), 4), " gradient norm",
          np.sqrt((G1 ** 2).sum() + (g2 ** 2).sum()))
    print("c = X'y/m =", c, " |c| =", round(np.linalg.norm(c), 4))
    print("Hessian eigenvalues at zero:", np.linalg.eigvalsh(hessian(Z1, z2)))

    # Equal hidden units stay equal
    E1 = np.array([W1[0], W1[0]])
    e2 = np.r_[w2[0], w2[1], w2[1]]
    Q1, q2, h = train(eta, T, W1=E1, w2=e2)
    print("equal start: rows equal at the end:", np.allclose(Q1[0], Q1[1]),
          f" L({T}) = {h[-1]:.6f}")
    print("W1 at the end:\n", Q1, "\nw2 at the end:", q2)
