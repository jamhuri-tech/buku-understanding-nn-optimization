"""Chapter 1: one full training run of gradient descent on the students."""
import numpy as np
from students import X, y, m, loss, grad, H, w_star, gd

eta = 0.03
w = np.zeros(3)
print("step 1: g0 =", grad(w).round(3))
w1 = w - eta * grad(w)
print("        w1 =", w1.round(4), " L =", round(loss(w1), 3))
r1 = X @ w1 - y
print("step 2: residuals r1 =", r1.round(3))
g1 = grad(w1)
print("        g1 =", g1.round(3))
w2 = w1 - eta * g1
print("        w2 =", w2.round(4), " L =", round(loss(w2), 3))
print("first-order prediction of L(w1)-L(0):",
      round(-eta * grad(w) @ grad(w), 2), " true:", round(loss(w1) - loss(w), 2))

print("\n t     w0      w1      w2        L     |g|")
for t, wt in enumerate(gd(eta, 10)):
    print(f"{t:2d} {wt[0]:7.3f} {wt[1]:7.3f} {wt[2]:7.3f} {loss(wt):9.3f}"
          f" {np.linalg.norm(grad(wt)):8.3f}")
for T in (100, 1000, 10000, 50000):
    wt = gd(eta, T)[-1]
    print(f"after {T:5d} steps: w = {wt.round(3)}, L = {loss(wt):.3f}")
print("w* =", w_star.round(3), " L* =", round(loss(w_star), 3))

lam, Q = np.linalg.eigh(H)
print("\nfactor per step along q_j, 1 - eta*lambda_j:", (1 - eta * lam).round(5))
print("\nL after 10 steps for several eta:")
for e in (0.01, 0.02, 0.03, 0.034, 0.036):
    print(f"  eta = {e}: L10 = {loss(gd(e, 10)[-1]):.3f}")
xnew = np.array([1, 2.0, 7.0])
w10k = gd(eta, 10000)[-1]
print("\nnew student (2 h, 7 h): w10 ->", round(xnew @ gd(eta, 10)[-1], 2),
      " w10000 ->", round(xnew @ w10k, 2), " w* ->", round(xnew @ w_star, 2))
