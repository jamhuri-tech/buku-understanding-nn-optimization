"""Chapter 3: steepest descent on the students."""
import numpy as np
from students import X, y, m, x1, x2, w_star, loss

Xc = np.column_stack([np.ones(m), x1 - x1.mean(), x2 - x2.mean()])
Xs = np.column_stack([np.ones(m), (x1 - x1.mean()) / x1.std(),
                      (x2 - x2.mean()) / x2.std()])
Lstar = loss(w_star)


def make(A):
    H = A.T @ A / m
    ws = np.linalg.solve(H, A.T @ y / m)
    L = lambda w: (A @ w - y) @ (A @ w - y) / (2 * m)
    g = lambda w: A.T @ (A @ w - y) / m
    return H, ws, L, g


def run(A, rule, target, maxit=10**6):
    """Steps until L <= target; rule(H, g) returns the step size."""
    H, ws, L, g = make(A)
    w = np.zeros(3)
    for t in range(maxit):
        if L(w) <= target:
            return t
        gw = g(w)
        w = w - rule(H, gw) * gw
    return None


exact = lambda H, g: (g @ g) / (g @ H @ g)
target = Lstar + 0.01
print(f"L* = {Lstar:.4f}, L(0) = {loss(np.zeros(3)):.4f}, target = {target:.4f}")
print("features    mu       beta     kappa   1/beta   2/(mu+beta)"
      "  rate1  rate2")
for name, A in (("raw", X), ("centred", Xc), ("standard", Xs)):
    H, *_ = make(A)
    lam = np.linalg.eigvalsh(H)
    mu, be = lam[0], lam[-1]
    k = be / mu
    print(f"{name:9s} {mu:8.4f} {be:8.4f} {k:9.2f} {1/be:8.4f} {2/(mu+be):8.4f}"
          f"  {1-mu/be:.4f} {(k-1)/(k+1):.4f}")
print("\nsteps to reach the target loss")
print("features   eta=1/beta  eta=2/(mu+beta)  exact   kappa*ln  r-bound")
for name, A in (("raw", X), ("centred", Xc), ("standard", Xs)):
    H, *_ = make(A)
    lam = np.linalg.eigvalsh(H)
    mu, be = lam[0], lam[-1]
    k = be / mu
    n1 = run(A, lambda H, g: 1 / be, target)
    n2 = run(A, lambda H, g: 2 / (mu + be), target)
    n3 = run(A, exact, target)
    gap0 = loss(np.zeros(3)) - Lstar
    pred = np.log(gap0 / 0.01) / -np.log(1 - 1 / k)
    pred2 = np.log(gap0 / 0.01) / (-2 * np.log((k - 1) / (k + 1)))
    print(f"{name:9s} {n1:9d} {n2:12d} {n3:9d} {pred:10.1f} {pred2:8.1f}")
print(f"gap0 = {gap0:.2f}, ln(gap0/0.01) = {np.log(gap0/0.01):.3f}")

# exact line search, first steps by hand on raw features
H, ws, L, g = make(X)
w = np.zeros(3)
print("\nexact line search, raw features")
for t in range(3):
    gw = g(w)
    eta = exact(H, gw)
    print(f"t={t}: g = {gw.round(4)}, g'g = {gw@gw:.3f},"
          f" g'Hg = {gw@H@gw:.3f}, eta = {eta:.5f}")
    if t == 0:
        print("      H =", H.round(4).tolist(), " Hg =", (H @ gw).round(3))
    w = w - eta * gw
    print(f"      w = {w.round(4)}, L = {L(w):.4f}, g_new.g = {g(w)@gw:.2e}")
# fixed eta = 0.03 from Ch 1 after one step
print("fixed eta=0.03, one step: L =", round(L(-0.03 * g(np.zeros(3))), 4))

# Kantorovich bound per step on the loss gap, raw and centred
for name, A in (("raw", X), ("centred", Xc)):
    H, ws, L, g = make(A)
    lam = np.linalg.eigvalsh(H)
    k = lam[-1] / lam[0]
    w = np.zeros(3)
    ratios = []
    for t in range(200):
        if L(w) - Lstar < 1e-9:
            break
        gw = g(w)
        w2 = w - exact(H, gw) * gw
        ratios.append((L(w2) - Lstar) / (L(w) - Lstar))
        w = w2
    print(f"{name}: bound ((k-1)/(k+1))^2 = {((k-1)/(k+1))**2:.6f},"
          f" worst ratio seen = {max(ratios):.6f},"
          f" steps used = {len(ratios)}")

# standardised features, block 2x2 hand example
H, ws, L, g = make(Xs)
print("\nstandardised H =\n", H.round(4))
print("w*_s =", ws.round(4))
gw = g(np.zeros(3))
print("g(0) =", gw.round(4), " eta_exact =", round(exact(H, gw), 5))
print("eigenvalues", np.linalg.eigvalsh(H).round(4))
r = 0.4275
print(f"gap factor r^2 = {r*r:.4f}, ln(1/r^2) = {np.log(1/r/r):.4f},"
      f" bound t >= {np.log(gap0/0.01)/np.log(1/r/r):.2f}")
w1 = -1.0 * gw
print("eta = 1: w1 =", w1.round(4), " L =", round(L(w1), 4))
H, ws, L, g = make(X)
print("\neta = 0.03 (Ch 1), raw: steps to target =",
      run(X, lambda H, g: 0.03, target))

# zigzag on the slice w0 = w0*, weights (w1, w2), from zero
print("\nslice w0 = w0*: exact line search on (w1, w2) from (0, 8)")
for name, A in (("raw", X), ("standard", Xs)):
    H, ws, L, g = make(A)
    S, c = H[1:, 1:], ws[1:]
    lam = np.linalg.eigvalsh(S)
    k = lam[1] / lam[0]
    v, steps, prev = np.array([0.0, 8.0]), 0, None
    while (v - c) @ S @ (v - c) / 2 > 0.01:
        gv = S @ (v - c)
        if prev is not None and steps < 3:
            print(f"  step {steps}: g_t . g_(t-1) = {gv @ prev:.1e}")
        v = v - (gv @ gv) / (gv @ S @ gv) * gv
        prev, steps = gv, steps + 1
    print(f"{name}: S = {S.round(4).tolist()}, eigenvalues {lam.round(4)},"
          f" kappa {k:.2f}, r = {(k-1)/(k+1):.4f}, steps to gap 0.01: {steps}")
