from tabnanny import verbose
import numpy as np
import nimsmo

from kernel import rbf_kernel as kernel
import rendered
from smooth_max import get_maxfun


def solve_system(k_p, a, s, g, h, lmbda, max_asum):
    idx_p = h > 0.0
    idx_z = ~idx_p
    da = np.zeros_like(a)
    da_z = a[idx_z] + g[idx_z]
    da[idx_z] = da_z
    h_p = h[idx_p]
    if np.isinf(max_asum):
        d_p, db = rendered.solve(
            k_p[:, idx_p] / lmbda + np.diag(1.0 / h_p),
            np.ones_like(h_p),
            (a[idx_p] + g[idx_p]) / h_p - k_p[:, idx_z].dot(da_z) / lmbda,
            a[idx_p].sum() - g[idx_z].sum(),
        )
        da[idx_p] = d_p
        return da, db, 0.0
    else:
        d_p, (db, dc) = rendered.solve(
            k_p[:, idx_p] / lmbda + np.diag(1.0 / h_p),
            (np.ones_like(h_p), s[idx_p]),
            (a[idx_p] + g[idx_p]) / h_p - k_p[:, idx_z].dot(da_z) / lmbda,
            (
                a[idx_p].sum() - g[idx_z].sum(),
                a.dot(s) - max_asum - s[idx_z].dot(da_z)
            ),
        )
        da[idx_p] = d_p
        return da, db, dc


def descent_step(z, x, y, ka, obj, f, g, r, params, direction='newton'):
    a, b, c = z
    m, dm, ddm = get_maxfun(params['smoothingParam'], params['maxfunDegree'])
    lmbda = params['lmbda']
    gamma = params['gamma']
    s = y
    max_asum = params.get('maxAsum', float('inf'))
    has_asum = ~np.isinf(max_asum)
    tol = params['tolNewton']

    if direction == 'newton':
        # compute Newton direction
        h = ddm(f)
        idx_p = h > 0
        k_p = kernel(x[idx_p], x, gamma)
        dn_a, dn_b, dn_c = solve_system(k_p, a, s, g, h, lmbda, max_asum)
    elif direction == 'gradient':
        # use preconditioned gradient
        dn_a, dn_b = r, lmbda * g.sum()
        if has_asum:
            dn_c = lmbda * (g.dot(s) + max_asum)
        else:
            dn_c = 0.0
    else:
        raise ValueError('unknown value for direction')

    # compute predicted decrease
    idx_d = np.abs(dn_a) > tol
    idx_r = np.abs(r) > tol
    dgrad = kernel(
        x[idx_d], x[idx_r], gamma
    ).dot(r[idx_r]).dot(dn_a[idx_d]) / lmbda
    dgrad += dn_b * g.sum()
    if has_asum:
        dgrad += dn_c * (g.dot(s) + max_asum)
    if dgrad < 0.0:
        if verbose != 0:
            print(f'no ascent direction: {dgrad} <= 0.0')
        return

    # find step size
    t = 1.0
    for _ in range(params.get('maxStepsBack', 10)):
        an = a - t * dn_a
        bn = b - t * dn_b
        cn = c - t * dn_c

        kan = ka - t * kernel(x, x[idx_d], gamma).dot(dn_a[idx_d]) / lmbda
        decn = kan + bn + s * cn
        fn = params['shift'] - y * decn
        valn = m(fn)
        objn = 0.5 * kan.dot(an)
        objn += valn.sum()
        if has_asum:
            objn += cn * max_asum

        if params.get('verbose', 0) != 0:
            print(f'  [{_+1}] objective value: {objn}')

        if objn < obj - params.get('sigma', 0.01) * t * dgrad + tol:
            break
        t *= params.get('beta', 0.1)
    else:
        return
    return (an, bn, cn), kan


def solve(x, y, kernel, params, z0=None):
    gamma = params['gamma']
    lmbda = params['lmbda']
    verbose = params.get('verbose', 0)
    max_asum = params.get('maxAsum', float('inf'))
    has_asum = ~np.isinf(max_asum)

    if z0 is None:
        n = x.shape[0]
        z = np.zeros(n), 0.0, 0.0
        ka = np.zeros(n)
    else:
        z = z0
        a = z[0]
        idx_sv = a != 0
        x_sv = x[idx_sv]
        k = kernel(x, x_sv, gamma)
        ka = k.dot(a[idx_sv]) / lmbda

    s = y
    m, dm, ddm = get_maxfun(params['smoothingParam'], params['maxfunDegree'])
    recompute_ka = False
    fresh_ka = True

    for _ in range(params.get('maxSteps', 1000)):
        a, b, c = z

        if recompute_ka:
            if verbose != 0:
                print('  ... recompute ka')
            idx_sv = a != 0
            ka = kernel(x, x[idx_sv], gamma).dot(a[idx_sv]) / lmbda
            recompute_ka = False
            fresh_ka = True

        # compute gradient and rhs
        dec = ka + b + s * c
        f = params['shift'] - y * dec
        val = m(f)
        obj = 0.5 * ka.dot(a)
        obj += val.sum()
        if has_asum:
            obj += c * max_asum
        g = -y * dm(f)
        r = a + g
        res = np.mean(np.array(r) ** 2)
        res += g.sum() ** 2
        if has_asum:
            res += (g.dot(s) + max_asum) ** 2
        if verbose != 0:
            print(f'[{_+1}] objective value: {obj}, residual: {res}')
        if res < params['tolNewton']:
            if fresh_ka:
                break
            recompute_ka = True
            continue

        # try Newton step
        result = descent_step(z, x, y, ka, obj, f, g, r, params)
        if result is not None:
            z, ka = result
            fresh_ka = False
            continue

        if fresh_ka:
            result = descent_step(
                z, x, y, ka, obj, f, g, r, params,
                direction='gradient',
            )
            if result is not None:
                z, ka = result
                fresh_ka = False
                continue
            raise RuntimeError('step failed')
        recompute_ka = True
    else:
        return

    return dict(a=a, b=b, c=c, ka=ka, g=g)


def solve_twostep(x, y, kernel, params):
    result = nimsmo.solve(
        x.tolist(), y.tolist(),
        params['lmbda'], params['gamma'],
        smoothingParam=params['smoothingParam'],
        maxAsum=params.get('maxAsum', float('inf')),
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

    gamma = 0.2

    params = dict(
        lmbda=1e-9,
        gamma=1.0,
        smoothingParam=0.1,
        shift=1.0,
        verbose=1000,
        tolSmo=1e-1,
        tolNewton=1e-12,
        maxfunDegree='2',
    )

    result = solve_twostep(gamma * x, y, kernel, params)
    a = np.array(result['a'])
    b = result['b']
    c = result['c']

    # check optimality conditions
    maxfun = get_maxfun(params['smoothingParam'], params['maxfunDegree'])
    m, dm, ddm = maxfun
    idx_sv = a != 0
    k = kernel(gamma * x, gamma * x[idx_sv])
    ka = k.dot(a[idx_sv]) / params['lmbda']

    s = y
    dec = ka + b + s * c
    f = params['shift'] - y * dec
    val = m(f)
    obj = 0.5 * ka.dot(a) + val.sum()
    g = -y * dm(f)
    r = a + g

    print(r.dot(r), a.sum())
