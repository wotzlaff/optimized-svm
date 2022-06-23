import numpy as np
from scipy.linalg import cho_factor, cho_solve


def solve(qm, qv, r, r0):
    if qm.size == 0:
        if type(qv) is tuple:
            return np.zeros_like(r), tuple([0.0] * len(r0))
        else:
            return np.zeros_like(r), 0.0
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
