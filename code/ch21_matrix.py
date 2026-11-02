"""Chapter 21: matrix preconditioners. Kronecker products, Shampoo
(Gupta, Koren and Singer 2018), SOAP (Vyas et al. 2025) and Muon
(Jordan et al. 2024) on the 2-2-1 network."""
import numpy as np
from net221 import init, loss, backward, Xs, ys
from ch10_perweight import batches, LIN, first
from ch14_adam import OM1, OM2, ZETA

np.set_printoptions(precision=4, suppress=True)
W10, w20 = init()
NS5 = (3.4445, -4.7750, 2.0315)        # quintic coefficients of the Muon post


def vec(A):
    """Stack the columns of A into one column (column-major)."""
    return A.reshape(-1, order="F")


def mpow(S, p):
    """Power of a symmetric positive definite matrix by eigendecomposition."""
    lam, Q = np.linalg.eigh(S)
    return (Q * lam ** p) @ Q.T


def ortho(M):
    """U V^T from the reduced SVD of M."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def newton_schulz(M, steps, coef=(1.5, -0.5, 0.0)):
    """X <- aX + b (X X^T) X + c (X X^T)^2 X from X = M/||M||_F."""
    a, b, c = coef
    X = M / np.linalg.norm(M)
    out = [X]
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
        out.append(X)
    return out


def ns_scalar(s, steps, coef=(1.5, -0.5, 0.0)):
    a, b, c = coef
    out = [s]
    for _ in range(steps):
        s = a * s + b * s ** 3 + c * s ** 5
        out.append(s)
    return np.array(out)


# --------------------------- optimisers ----------------------------------
def adam_step(P, g, st, t, eta):
    m, v = st.get("m", 0 * P), st.get("v", 0 * P)
    m = OM1 * m + (1 - OM1) * g
    v = OM2 * v + (1 - OM2) * g * g
    st["m"], st["v"] = m, v
    return P - eta * (m / (1 - OM1 ** t)) / (np.sqrt(v / (1 - OM2 ** t)) + ZETA)


def shampoo_step(P, g, st, t, eta, eps=1e-4):
    """Algorithm 1 of the paper; a vector is treated as a 1 x n matrix."""
    G = np.atleast_2d(g)
    Lf = st.get("L", eps * np.eye(G.shape[0])) + G @ G.T
    Rf = st.get("R", eps * np.eye(G.shape[1])) + G.T @ G
    st["L"], st["R"] = Lf, Rf
    return P - eta * (mpow(Lf, -0.25) @ G @ mpow(Rf, -0.25)).reshape(P.shape)


def soap_step(P, g, st, t, eta, f=10, b1=OM1, b2=OM2):
    """Algorithm 3 of the paper: Adam in the eigenbasis of L and R."""
    G = np.atleast_2d(g)
    d1, d2 = G.shape
    if t == 1:
        st.update(L=np.zeros((d1, d1)), R=np.zeros((d2, d2)),
                  QL=np.linalg.eigh(G @ G.T)[1], QR=np.linalg.eigh(G.T @ G)[1],
                  M=0 * G, V=0 * G)
    QL, QR = st["QL"], st["QR"]
    Gr = QL.T @ G @ QR
    st["M"] = b1 * st["M"] + (1 - b1) * G
    Mr = QL.T @ st["M"] @ QR
    st["V"] = b2 * st["V"] + (1 - b2) * Gr * Gr
    Nr = (Mr / (1 - b1 ** t)) / (np.sqrt(st["V"] / (1 - b2 ** t)) + ZETA)
    P = P - eta * (QL @ Nr @ QR.T).reshape(P.shape)
    st["L"] = b2 * st["L"] + (1 - b2) * G @ G.T
    st["R"] = b2 * st["R"] + (1 - b2) * G.T @ G
    if t % f == 0:
        st["QL"] = np.linalg.qr(st["L"] @ QL)[0]     # one power step + QR
        st["QR"] = np.linalg.qr(st["R"] @ QR)[0]
    return P


def muon_step(P, g, st, t, eta, gam=0.95, steps=5):
    """Nesterov momentum, then quintic Newton-Schulz orthogonalisation."""
    B = gam * st.get("B", 0 * g) + g
    st["B"] = B
    return P - eta * newton_schulz(g + gam * B, steps, NS5)[-1]


STEP = {"adam": adam_step, "shampoo": shampoo_step, "soap": soap_step,
        "muon": muon_step}


def run(kind, eta, steps=2000, b=None, eta2=None, seed=9):
    """kind sets the method for W1; the output layer uses the same method,
    except Muon, which pairs with Adam at rate eta2 on w2."""
    W1, w2 = W10.copy(), w20.copy()
    s1, s2 = {}, {}
    k2 = "adam" if kind == "muon" else kind
    e2 = eta2 if kind == "muon" else eta
    gen = batches(b, seed) if b else None
    hist = [loss(W1, w2)]
    for t in range(1, steps + 1):
        idx = next(gen) if gen else slice(None)
        G1, g2 = backward(W1, w2, Xs[idx], ys[idx])
        W1 = STEP[kind](W1, G1, s1, t, eta)
        w2 = STEP[k2](w2, g2, s2, t, e2)
        hist.append(loss(W1, w2))
    return np.array(hist)


MEMORY = {"Adam": 2 * 6 + 2 * 3,
          "Shampoo": (4 + 9) + (1 + 9),
          "SOAP": (2 * 4 + 2 * 9 + 2 * 6) + (2 * 1 + 2 * 9 + 2 * 3),
          "Muon + Adam": 6 + 2 * 3,
          "full AdaGrad": 81}


def to_star(h):
    i = first(h, LIN)
    return "never" if i is None else i


# --------------------------- printing ------------------------------------
def hand_kron():
    A = np.array([[1.0, 2], [0, 1]])
    B = np.array([[2.0, 0], [1, 3]])
    print("A kron B =\n", np.kron(A, B))
    Lm = np.array([[2.0, 1], [0, 1]])
    G = np.array([[1.0, -1], [2, 0]])
    Rm = np.array([[1.0, 0], [1, 2]])
    print("L G R =\n", Lm @ G @ Rm, " vec =", vec(Lm @ G @ Rm))
    print("R^T kron L =\n", np.kron(Rm.T, Lm))
    print("(R^T kron L) vec G =", np.kron(Rm.T, Lm) @ vec(G))


def hand_shampoo(eps=1e-4):
    G = backward(W10, w20)[0]
    print("G =\n", G)
    Lf = G @ G.T + eps * np.eye(2)
    Rf = G.T @ G + eps * np.eye(3)
    print("L =\n", Lf)
    print("R =\n", Rf)
    for nm, S in (("L", Lf), ("R", Rf)):
        lam, Q = np.linalg.eigh(S)
        Q = Q * np.sign(Q[np.argmax(abs(Q), axis=0), range(len(lam))])
        print(nm, "eigenvalues", np.array2string(lam, formatter={"float": lambda x: f"{x:.4e}"}),
              " ^(-1/4)", lam ** -0.25)
        print(nm, "eigenvectors (columns)\n", Q)
        print(nm, "^(-1/4) =\n", mpow(S, -0.25))
    S = mpow(Lf, -0.25) @ G @ mpow(Rf, -0.25)
    print("L^(-1/4) G R^(-1/4) =\n", S)
    print("its singular values", np.linalg.svd(S, compute_uv=False))
    print("Frobenius norms: G", np.linalg.norm(G), " step", np.linalg.norm(S))


def hand_svd():
    G = backward(W10, w20)[0]
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    sg = np.sign(U[0])
    U, Vt = U * sg, Vt * sg[:, None]
    print("U =\n", U)
    print("singular values", s)
    print("V^T =\n", Vt)
    O = U @ Vt
    print("U V^T =\n", O)
    print("||G||_F =", np.linalg.norm(G), " start sigma", s / np.linalg.norm(G))
    Xs_ = newton_schulz(G, 3)
    for k, X in enumerate(Xs_):
        print(f"cubic X_{k} =\n", X, " sv", np.linalg.svd(X, compute_uv=False))
    print("cubic sigma path", ns_scalar(s / np.linalg.norm(G), 8))
    print("quintic sigma path", ns_scalar(s / np.linalg.norm(G), 5, NS5))
    X5 = newton_schulz(G, 5, NS5)[-1]
    print("quintic X_5 =\n", X5, " sv", np.linalg.svd(X5, compute_uv=False))


ETAS = (0.01, 0.03, 0.1, 0.3, 1.0)


def hand_compare(eta=0.1, zeta=1e-4):
    """Numbers behind the SVD route and one step on W1 with each direction."""
    G = backward(W10, w20)[0]
    S = G @ G.T
    print(f"trace {np.trace(S):.6f}  det {np.linalg.det(S):.4e}")
    lam = np.linalg.eigvalsh(S)
    s = np.sqrt(lam[::-1])
    print(f"sigma^2 {lam[1]:.4e} {lam[0]:.4e}  sigma {s[0]:.6f} {s[1]:.6f}")
    c = np.cross(G[0], G[1])
    print("cross product of the rows", c / np.linalg.norm(c))
    print("Shampoo singular values sigma/sqrt(sigma^2+zeta)",
          s / np.sqrt(s ** 2 + zeta))
    Sh = mpow(S + zeta * np.eye(2), -0.25) @ G @ mpow(G.T @ G + zeta * np.eye(3), -0.25)
    print(f"L(w0) = {loss(W10, w20):.4f}")
    for nm, D in (("G", G), ("Shampoo", Sh), ("UV^T", ortho(G))):
        print(f"  step with {nm:8s} ||D||_F {np.linalg.norm(D):.4f}"
              f"  <G,D> {np.sum(G * D):.4f}  L {loss(W10 - eta * D, w20):.4f}")


def grids():
    print("full batch, steps to L* within 2000 steps, eta =", ETAS)
    for kind in STEP:
        row = [to_star(run(kind, eta, eta2=ETA2)) for eta in ETAS]
        print(f"  {kind:8s}" + "".join(f"{r!s:>7}" for r in row))


ETA2 = 0.1                             # Adam on w2 next to Muon
BEST = {"adam": 0.1, "shampoo": 0.3, "soap": 0.3, "muon": 0.03}


def table():
    print("method   b  to L*  L(500)     L(2000)    mean last 200")
    out = {}
    for b in (None, 2):
        for kind, eta in BEST.items():
            h = run(kind, eta, b=b, eta2=ETA2)
            out[(kind, b)] = h
            print(f"  {kind:8s} {b or 8}  {to_star(h)!s:>5}  {h[500]:.3e}  "
                  f"{h[-1]:.3e}  {h[-200:].mean():.3e}")
    return out


if __name__ == "__main__":
    print("== Kronecker product and the vec identity ==")
    hand_kron()
    print("== One Shampoo preconditioner on G at the start ==")
    hand_shampoo()
    print("== SVD and Newton-Schulz ==")
    hand_svd()
    hand_compare()
    print("== memory (stored numbers besides weights and gradient) ==")
    for k, v in MEMORY.items():
        print(f"  {k:13s} {v}")
    print("== grids ==")
    grids()
    print("== table ==")
    table()
