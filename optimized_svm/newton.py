import numpy as np
import nimsmo

from kernel import rbf_kernel as kernel
from rendered import solve_rendered
from smooth_max import get_maxfun


def solve_system(k_p, a, g, h, lmbda):
    idx_p = h > 0.0
    idx_z = ~idx_p
    d = np.zeros_like(a)
    d_z = a[idx_z] + g[idx_z]
    d[idx_z] = d_z
    d_p, d_0 = solve_rendered(
        k_p[:, idx_p], h[idx_p],
        a[idx_p] + g[idx_p] - h[idx_p] * k_p[:, idx_z].dot(d_z) / lmbda,
        a.sum() - d_z.sum(),
        lmbda,
    )
    d[idx_p] = d_p
    return d, d_0


def descent_step(
    a, b, c,
    x, y, ka, obj, f, g, r,
    params,
    direction,
):
    m, dm, ddm = get_maxfun(params['smoothingParam'], params['maxfunDegree'])
    lmbda = params['lmbda']
    gamma = params['gamma']

    if direction == 'newton':
        # compute Newton direction
        h = ddm(f)
        idx_p = h > 0
        k_p = kernel(x[idx_p], x, gamma)
        dn, dn_b = solve_system(k_p, a, g, h, lmbda)
        dn_c = 0.0
    elif direction == 'gradient':
        # use preconditioned gradient
        dn, dn_b = r, lmbda * g.sum()
        dn_c = 0.0
    else:
        raise ValueError('unknown value for direction')

    # compute predicted decrease
    idx_d = np.abs(dn) > params['tol']
    idx_r = np.abs(r) > params['tol']
    dgrad = kernel(
        x[idx_d], x[idx_r], gamma
    ).dot(r[idx_r]).dot(dn[idx_d]) / lmbda + dn_b * g.sum()
    assert dgrad > 0.0, 'no ascent direction'

    # find step size
    t = 1.0
    for _ in range(params.get('maxStepsBack', 10)):
        an = a - t * dn
        bn = b - t * dn_b
        cn = c - t * dn_c

        kan = ka - t * kernel(x, x[idx_d], gamma).dot(dn[idx_d]) / lmbda
        decn = kan + bn
        fn = params['shift'] - y * decn
        valn = m(fn)
        objn = 0.5 * kan.dot(an) + valn.sum()
        if params['verbose'] != 0:
            print(f'  [{_+1}] objective value: {objn}')

        if objn < obj - params.get('sigma', 0.1) * t * dgrad + params['tol']:
            break
        t *= params.get('beta', 0.5)
    else:
        return
    return an, bn, cn, kan


def solve(x, y, kernel, params, z0=None):
    gamma = params['gamma']
    lmbda = params['lmbda']
    verbose = params['verbose']

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

    m, dm, ddm = get_maxfun(params['smoothingParam'], params['maxfunDegree'])
    recompute_ka = False
    fresh_ka = True

    for _ in range(params.get('maxSteps', 1000)):
        if recompute_ka:
            if verbose != 0:
                print('  ... recompute ka')
            idx_sv = a != 0
            ka = kernel(x, x[idx_sv], gamma).dot(a[idx_sv]) / lmbda
            recompute_ka = False
            fresh_ka = True

        # compute gradient and rhs
        dec = ka + b
        f = params['shift'] - y * dec
        val = m(f)
        obj = 0.5 * ka.dot(a) + val.sum()
        g = -y * dm(f)
        r = a + g
        res = np.mean(np.array(r)**2) + g.sum() ** 2
        if verbose != 0:
            print(f'[{_+1}] objective value: {obj}, residual: {res}')
        if res < params['tol']:
            if fresh_ka:
                break
            recompute_ka = True
            continue

        # try Newton step
        result = descent_step(
            a, b, c, x, y, ka, obj, f, g, r, params,
            direction='newton',
        )
        if result is not None:
            a, b, c, ka = result
            fresh_ka = False
            continue

        if fresh_ka:
            result = descent_step(
                a, b, c, x, y, ka, obj, f, g, r, params,
                direction='gradient',
            )
            if result is not None:
                a, b, c, ka = result
                fresh_ka = False
                continue
            raise RuntimeError('step failed')
        recompute_ka = True

    return dict(a=a, b=b, c=c, ka=ka, g=g)


def solve_twostep(x, y, kernel, params):
    lmbda = params['lmbda']
    gamma = params['gamma']
    result = nimsmo.solve(
        x.tolist(), y.tolist(),
        lmbda, gamma,
        smoothingParam=params['smoothingParam'],
        shift=params['shift'],
        tolViolation=params.get('tolSmo', 0.1),
        verbose=params.get('verbose', 0),
        shrinkingPeriod=params.get('shrinkingPeriod', 100),
    )
    a = np.array(result['a'])
    b = result['b']
    c = result['c']
    return solve(x, y, kernel, params, z0=(a, b, c))


if __name__ == '__main__':
    np.random.seed(42)
    n, nft = 1000, 10
    x = np.random.rand(n, nft)
    yr = np.sin(np.pi * x).sum(axis=1)
    y = np.where(yr > np.median(yr), 1.0, -1.0)

    params = dict(
        lmbda=0.1,
        gamma=0.5,
        smoothingParam=0.1,
        shift=1.0,
        verbose=1,
        tolSmo=1e-1,
        tol=1e-12,
        maxfunDegree='2',
    )

    result = solve_twostep(x, y, kernel, params)
    a = np.array(result['a'])
    b = result['b']
    c = result['c']

    # check optimality conditions
    maxfun = get_maxfun(params['smoothingParam'], params['maxfunDegree'])
    m, dm, ddm = maxfun
    idx_sv = a != 0
    k = kernel(x, x[idx_sv], params['gamma'])
    ka = k.dot(a[idx_sv]) / params['lmbda']

    dec = ka + b
    f = params['shift'] - y * dec
    val = m(f)
    obj = 0.5 * ka.dot(a) + val.sum()
    g = -y * dm(f)
    r = a + g

    print(r.dot(r), a.sum())
