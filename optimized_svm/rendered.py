import numpy as np
from scipy.linalg import cho_factor, cho_solve


def solve_rendered(k_p, h_p, rhs_p, rhs_0, lmbda):
    qm = k_p / lmbda + np.diag(1.0 / h_p)
    a_p = rhs_p / h_p

    qm_cho = cho_factor(qm, overwrite_a=True)
    p_p = cho_solve(qm_cho, a_p)
    q_p = cho_solve(qm_cho, np.ones_like(rhs_p))

    u_0 = (p_p.sum() - rhs_0) / q_p.sum()
    u_p = p_p - u_0 * q_p
    return u_p, u_0


def solve_rendered_nu(k_p, h_p, s_p, rhs_p, rhs_1, rhs_s, lmbda):
    qm = k_p / lmbda + np.diag(1.0 / h_p)
    a_p = rhs_p / h_p

    qm_cho = cho_factor(qm, overwrite_a=True)
    p_p = cho_solve(qm_cho, a_p)
    q1_p = cho_solve(qm_cho, np.ones_like(rhs_p))
    qs_p = cho_solve(qm_cho, s_p)

    c11 = q1_p.sum()
    c1s = qs_p.sum()
    css = qs_p.dot(s_p)
    det = c11 * css - c1s ** 2
    r1 = q1_p.dot(a_p) - rhs_1
    rs = qs_p.dot(a_p) - rhs_s

    d1 = (r1 * css - rs * c1s) / det
    ds = (rs * c11 - r1 * c1s) / det

    d_p = p_p - d1 * q1_p - ds * qs_p
    return d_p, d1, ds


def solve_rendered_transposed(k_p, h_p, lhs_p, lhs_0, lmbda):
    qm = k_p / lmbda + np.diag(1.0 / h_p)

    qm_cho = cho_factor(qm, overwrite_a=True)
    p_p = cho_solve(qm_cho, lhs_p)
    q_p = cho_solve(qm_cho, np.ones_like(lhs_p))

    u_0 = (p_p.sum() - lhs_0) / q_p.sum()
    # return scaled hu_p = h_p * u_p
    hu_p = p_p - u_0 * q_p
    return hu_p, u_0
