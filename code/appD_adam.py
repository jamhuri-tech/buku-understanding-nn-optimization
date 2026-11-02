"""Appendix D: two Adam steps on the eight students, as in the
hand example of Chapter 14, written with NumPy only."""
import numpy as np
from students import grad, loss

eta, om1, om2, zeta = 0.01, 0.9, 0.999, 1e-8
w = np.zeros(3)
m = np.zeros(3)
r = np.zeros(3)
for t in range(2):
    g = grad(w)
    m = om1 * m + (1 - om1) * g
    r = om2 * r + (1 - om2) * g * g
    m_hat = m / (1 - om1 ** (t + 1))
    r_hat = r / (1 - om2 ** (t + 1))
    w = w - eta * m_hat / (np.sqrt(r_hat) + zeta)
    print(f"t = {t}  m = {np.round(m, 3)}")
    print(f"       r = {np.round(r, 4)}")
    print(f"       w = {np.round(w, 7)}  L = {loss(w):.2f}")
