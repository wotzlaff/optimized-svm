import numpy as np
from scipy.linalg import cho_factor, cho_solve
import nimsmo

from smooth_max import maxfun as get_maxfun


def solve_system(k_p, a, g, h, lmbda):
    idx_p = h > 0.0
    idx_z = ~idx_p
    d = np.zeros_like(a)
    d_z = a[idx_z] + g[idx_z]
    d[idx_z] = d_z

    if d_z.size == d.size:
        return d, 0.0

    an_p = (a[idx_p] + g[idx_p]) / h[idx_p] - k_p[:, idx_z].dot(d_z) / lmbda
    an_0 = a[idx_p].sum() - g[idx_z].sum()
    qm = np.diag(1.0 / h[idx_p]) + k_p[:, idx_p] / lmbda
    qm_cho = cho_factor(qm, overwrite_a=True)
    p_p = cho_solve(qm_cho, an_p)
    q_p = cho_solve(qm_cho, np.ones_like(an_p))
    d_0 = (p_p.sum() - an_0) / q_p.sum()
    d_p = p_p - d_0 * q_p
    d[idx_p] = d_p
    return d, d_0


def descent_step(
    a, b,
    x, y, ka, obj, f, g, r,
    maxfun,
    params,
    direction,
):
    m, dm, ddm = maxfun
    lmbda = params['lmbda']
    gamma = params['gamma']

    if direction == 'newton':
        # compute Newton direction
        h = ddm(f)
        idx_p = h > 0
        k_p = kernel(x[idx_p], x, gamma)
        dn, dn_0 = solve_system(k_p, a, g, h, lmbda)
    elif direction == 'gradient':
        # use preconditioned gradient
        dn, dn_0 = r, lmbda * g.sum()
    else:
        raise ValueError('unknown value for direction')

    # compute predicted decrease
    idx_d = np.abs(dn) > params['tol']
    idx_r = np.abs(r) > params['tol']
    dgrad = kernel(
        x[idx_d], x[idx_r], gamma
    ).dot(r[idx_r]).dot(dn[idx_d]) / lmbda + dn_0 * g.sum()
    assert dgrad > 0.0, 'no ascent direction'

    # find step size
    t = 1.0
    for _ in range(params['max_steps_back']):
        an = a - t * dn
        bn = b - t * dn_0

        kan = ka - t * kernel(x, x[idx_d], gamma).dot(dn[idx_d]) / lmbda
        decn = kan + bn
        fn = params['shift'] - y * decn
        valn = m(fn)
        objn = 0.5 * kan.dot(an) + valn.sum()
        if params['verbose'] != 0:
            print(f'  [{_+1}] objective value: {objn}')

        if objn < obj - params['sigma'] * t * dgrad + params['tol']:
            break
        t *= params['beta']
    else:
        return
    return an, bn, kan


def solve(
    x, y,
    kernel,
    gamma, lmbda, shift,
    maxfun,
    z0=None,
    max_steps=10000,
    tol=1e-12,
    max_steps_back=10,
    sigma=0.1,
    beta=0.1,
    verbose=0,
):
    if z0 is None:
        n = x.shape[0]
        a = np.zeros(n)
        b = 0.0
        c = 0.0
        ka = np.zeros(n)
    else:
        a, b, c = z0
        idx_sv = a != 0
        x_sv = x[idx_sv]
        k = kernel(x, x_sv, gamma)
        ka = k.dot(a[idx_sv]) / lmbda

    params = dict(
        lmbda=lmbda, gamma=gamma, shift=shift,
        tol=tol,
        max_steps=max_steps, max_steps_back=max_steps_back,
        sigma=sigma, beta=beta,
        verbose=verbose,
    )

    m, dm, ddm = maxfun
    recompute_ka = False
    fresh_ka = True

    for _ in range(max_steps):
        if recompute_ka:
            if verbose != 0:
                print('  ... recompute ka')
            idx_sv = a != 0
            ka = kernel(x, x[idx_sv], gamma).dot(a[idx_sv]) / lmbda
            recompute_ka = False
            fresh_ka = True

        # compute gradient and rhs
        dec = ka + b
        f = shift - y * dec
        val = m(f)
        obj = 0.5 * ka.dot(a) + val.sum()
        g = -y * dm(f)
        r = a + g
        res = np.mean(np.array(r)**2) + g.sum() ** 2
        if verbose != 0:
            print(f'[{_+1}] objective value: {obj}, residual: {res}')
        if res < tol:
            break

        # try Newton step
        result = descent_step(
            a, b, x, y, ka, obj, f, g, r, maxfun, params,
            direction='newton',
        )
        if result is not None:
            a, b, ka = result
            fresh_ka = False
            continue

        if fresh_ka:
            result = descent_step(
                a, b, x, y, ka, obj, f, g, r, maxfun, params,
                direction='gradient',
            )
            if result is not None:
                a, b, ka = result
                fresh_ka = False
                continue
            raise RuntimeError('step failed')
        recompute_ka = True

    return dict(
        a=a, b=b, c=c,
        ka=ka, g=g,
    )


def kernel(x1, x2, gamma):
    x1sqr = (x1 * x1).sum(axis=1)
    x2sqr = (x2 * x2).sum(axis=1)
    d = x1sqr[:, None] + x2sqr[None, :] - 2.0 * x1.dot(x2.T)
    return np.exp(-gamma * d)


def solve_twostep(
    x, y,
    maxfun,
    lmbda, gamma,
    smoothingParam, shift,
    verbose=0,
    tolSmo=1e-1,
):
    result = nimsmo.solve(
        x.tolist(),
        y.tolist(),
        lmbda,
        gamma,
        smoothingParam=smoothingParam,
        shift=shift,
        tolViolation=tolSmo,
        verbose=verbose,
        shrinkingPeriod=100,
    )

    a = np.array(result['a'])
    b = result['b']
    c = result['c']
    z = a, b, c

    return solve(
        x, y, kernel,
        gamma, lmbda, shift,
        maxfun,
        z,
        verbose=verbose,
    )


if __name__ == '__main__':
    np.random.seed(42)
    n, nft = 100, 10
    x = np.random.rand(n, nft)
    yr = np.sin(np.pi * x).sum(axis=1)
    y = np.where(yr > np.median(yr), 1.0, -1.0)

    lmbda = 0.1
    gamma = 0.5
    smoothingParam = 0.1
    shift = 1.0
    maxAsum = 20.0

    result = nimsmo.solve(
        x.tolist(),
        y.tolist(),
        lmbda,
        gamma,
        maxAsum=maxAsum,
        smoothingParam=smoothingParam,
        shift=shift,
        tolViolation=1e-8,
        verbose=100,
        shrinkingPeriod=100,
    )

    a = np.array(result['a'])
    b = result['b']
    c = result['c']
    z = a, b, c

    # check optimality conditions
    m, dm, ddm = get_maxfun(smoothingParam, deg='2')
    idx_sv = a != 0
    k = kernel(x, x[idx_sv], gamma)
    ka = k.dot(a[idx_sv]) / lmbda

    dec = ka + b + y * c
    f = shift - y * dec
    val = m(f)
    obj = 0.5 * ka.dot(a) + val.sum()
    g = -y * dm(f)
    r = a + g

    print(r.dot(r), a.sum(), a.dot(y) - maxAsum)
