from tabnanny import verbose
import numpy as np
import nimsmo

from sample import data
from kernel import rbf_kernel as kernel
import rendered
from smooth_max import get_maxfun
import problem


def solve_system(prob, dec, a, g, lmbda, max_asum):
    h = prob.ddloss(dec)
    idx_p = h > 0.0
    k_p = prob.kernel(idx=idx_p).T
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
        s = prob.sign
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


def descent_step(prob, z, ka, dec, obj, g, r, direction='newton'):
    params = prob.params
    a, b, c = z
    lmbda = params['lmbda']
    max_asum = params.get('maxAsum', float('inf'))
    has_asum = ~np.isinf(max_asum)
    tol = params['tolNewton']

    if direction == 'newton':
        # compute Newton direction
        dn_a, dn_b, dn_c = solve_system(
            prob, dec,
            a, g, lmbda, max_asum,
        )
    elif direction == 'gradient':
        # use preconditioned gradient
        dn_a, dn_b = r, lmbda * g.sum()
        if has_asum:
            dn_c = lmbda * (g.dot(prob.sign) + max_asum)
        else:
            dn_c = 0.0
    else:
        raise ValueError('unknown value for direction')

    # compute predicted decrease
    idx_d = np.abs(dn_a) > tol
    idx_r = np.abs(r) > tol
    dgrad = prob.kernel(
        idx=idx_r, idx0=idx_d
    ).dot(r[idx_r]).dot(dn_a[idx_d]) / lmbda
    dgrad += dn_b * g.sum()
    if has_asum:
        dgrad += dn_c * (g.dot(prob.sign) + max_asum)
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
        zn = (an, bn, cn)

        kan = ka - t * prob.kernel(idx=idx_d).dot(dn_a[idx_d]) / lmbda
        objn = prob.objective(zn, kan)
        if params.get('verbose', 0) != 0:
            print(f'  [{_+1}] objective value: {objn}')

        if objn < obj - params.get('sigma', 0.01) * t * dgrad + tol:
            break
        t *= params.get('beta', 0.1)
    else:
        return
    return zn, kan


def solve(prob, z0=None):
    params = prob.params
    lmbda = params['lmbda']
    verbose = params.get('verbose', 0)
    max_asum = params.get('maxAsum', float('inf'))
    has_asum = ~np.isinf(max_asum)

    if z0 is None:
        n = prob.size
        z = np.zeros(n), 0.0, 0.0
        ka = np.zeros(n)
    else:
        z = z0
        a = z[0]
        assert a.size == prob.size
        idx_sv = a != 0
        ka = prob.kernel(idx=idx_sv).dot(a[idx_sv]) / lmbda

    recompute_ka = False
    fresh_ka = True

    for _ in range(params.get('maxSteps', 1000)):
        a, b, c = z

        if recompute_ka:
            if verbose != 0:
                print('  ... recompute ka')
            idx_sv = a != 0
            ka = prob.kernel(idx=idx_sv).dot(a[idx_sv]) / lmbda
            recompute_ka = False
            fresh_ka = True

        # compute gradient and rhs
        dec, obj = prob.decision_objective(z, ka)
        g = prob.dloss(dec)

        r = a + g
        res = np.mean(np.array(r) ** 2)
        res += g.sum() ** 2
        if has_asum:
            res += (g.dot(prob.sign) + max_asum) ** 2
        if verbose != 0:
            print(f'[{_+1}] objective value: {obj}, residual: {res}')
        if res < params['tolNewton']:
            if fresh_ka:
                break
            recompute_ka = True
            continue

        # try Newton step
        result = descent_step(prob, z, ka, dec, obj, g, r)
        if result is not None:
            z, ka = result
            fresh_ka = False
            continue

        if fresh_ka:
            result = descent_step(
                prob, z, ka, dec, obj, g, r,
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


def solve_twostep(prob):
    params = prob._params
    result = nimsmo.solveClassification(
        prob._x.tolist(), prob._y.tolist(),
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
    return solve(prob, z0=(a, b, c))


def main():
    x, y = data(1000, 10)

    gamma = 0.2

    params = dict(
        lmbda=1e-3,
        gamma=1.0,
        smoothingParam=0.1,
        shift=1.0,
        verbose=1000,
        tolSmo=1e-3,
        tolNewton=1e-12,
        maxfunDegree='2',
    )

    prob = problem.Classification(gamma * x, y, params)
    result = solve_twostep(prob)
    a = np.array(result['a'])
    b = result['b']
    c = result['c']

    dec, obj = prob.decision_objective((a, b, c))
    print(obj)
    g = prob.dloss(dec)
    r = a + g
    print(r.dot(r), a.sum())


if __name__ == '__main__':
    main()
