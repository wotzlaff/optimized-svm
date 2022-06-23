import numpy as np
from scipy.linalg import cho_factor, cho_solve


def solve(qm, qv, r, r0):
    q_cho = cho_factor(qm, overwrite_a=True)
    if type(qv) is tuple:
        a = np.stack(qv, axis=-1)
        qia = cho_solve(q_cho, a)
        u0 = np.linalg.solve(qia.T.dot(a), qia.T.dot(r) - r0)
        u = cho_solve(q_cho, r) - qia.dot(u0)
        return u, tuple(u0)
    else:
        qir = cho_solve(q_cho, r)
        qiqv = cho_solve(q_cho, qv)
        u0 = (qir.dot(qv) - r0) / qiqv.dot(qv)
        u = qir - u0 * qiqv
        return u, u0


def solve_rendered(k_p, h_p, rhs_p, rhs_0, lmbda):
    return solve(
        k_p / lmbda + np.diag(1.0 / h_p),
        np.ones_like(rhs_p),
        rhs_p / h_p,
        rhs_0,
    )


def solve_rendered_nu(k_p, h_p, s_p, rhs_p, rhs_1, rhs_s, lmbda):
    return solve(
        k_p / lmbda + np.diag(1.0 / h_p),
        (np.ones_like(rhs_p), s_p),
        rhs_p / h_p,
        (rhs_1, rhs_s),
    )


def solve_rendered_transposed(k_p, h_p, lhs_p, lhs_0, lmbda):
    # return scaled hu_p = h_p * u_p
    return solve(
        k_p / lmbda + np.diag(1.0 / h_p),
        np.ones_like(lhs_p),
        lhs_p,
        lhs_0,
    )


def solve_rendered_transposed_nu(k_p, h_p, s_p, lhs_p, lhs_1, lhs_s, lmbda):
    # return scaled hu_p = h_p * u_p
    return solve(
        k_p / lmbda + np.diag(1.0 / h_p),
        (np.ones_like(lhs_p), s_p),
        lhs_p,
        (lhs_1, lhs_s),
    )
