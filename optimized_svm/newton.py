import time
import numpy as np
import nimsmo

from .rendered import solve as solve_rendered
from .problem import Classification, Regression


def solve_system(prob, dec, a, g, lmbda, max_asum):
    h = prob.ddloss(dec)
    idx_p = h > 0.0
    k_p = prob.k(idx=idx_p).T
    idx_z = ~idx_p
    da = np.zeros_like(a)
    da_z = a[idx_z] + g[idx_z]
    da[idx_z] = da_z
    h_p = h[idx_p]
    if np.isinf(max_asum):
        d_p, db = solve_rendered(
            k_p[:, idx_p] / lmbda + np.diag(1.0 / h_p),
            np.ones_like(h_p),
            (a[idx_p] + g[idx_p]) / h_p - k_p[:, idx_z].dot(da_z) / lmbda,
            a[idx_p].sum() - g[idx_z].sum(),
        )
        da[idx_p] = d_p
        return da, db, 0.0
    else:
        s = prob.sign
        d_p, (db, dc) = solve_rendered(
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


def descent_step(
    prob, z, ka, dec, obj, g, r,
    direction='newton',
    final_step=False,
    params=dict(),
):
    a, b, c = z
    lmbda = prob.lmbda
    max_asum = prob.maxAsum
    has_asum = not np.isinf(max_asum)
    tol = params.get('tol', 1e-10)

    def get_gradient():
        # use preconditioned gradient
        dn_a, dn_b = r, lmbda * g.sum()
        if has_asum:
            dn_c = lmbda * (g.dot(prob.sign) + max_asum)
        else:
            dn_c = 0.0
        if final_step:
            # this should not happen
            return
        return dn_a, dn_b, dn_c

    if direction == 'newton':
        # compute Newton direction
        try:
            d = solve_system(prob, dec, a, g, lmbda, max_asum)
        except np.linalg.LinAlgError:
            d = get_gradient()
    elif direction == 'gradient':
        d = get_gradient()
    else:
        raise ValueError('unknown value for direction')
    if d is None and final_step:
        # this should not happen
        return z, ka
    dn_a, dn_b, dn_c = d

    # compute predicted decrease
    idx_d = np.abs(dn_a) > tol
    idx_r = np.abs(r) > tol
    dgrad = prob.k(
        idx=idx_r, idx0=idx_d
    ).dot(r[idx_r]).dot(dn_a[idx_d]) / lmbda
    dgrad += dn_b * g.sum()
    if has_asum:
        dgrad += dn_c * (g.dot(prob.sign) + max_asum)
    if dgrad < 0.0:
        if params.get('verbose', 0) > 0:
            print(f'no ascent direction: {dgrad} <= 0.0')
        return

    # find step size
    t = 1.0
    for _ in range(params.get('maxStepsBack', 10)):
        an = a - t * dn_a
        bn = b - t * dn_b
        cn = c - t * dn_c
        zn = (an, bn, cn)

        kan = ka - t * prob.k(idx=idx_d).dot(dn_a[idx_d]) / lmbda
        objn = prob.objective(zn, kan)
        dobj = objn - obj
        if params.get('verbose', 0) > 1:
            tag = dict(newton='N', gradient='G')[direction]
            print(f'[{tag} {_+1:3d}] f = {obj:10.5g}, ', end='')
            print(f'{"":16s} df = {dobj:9.4e}')

        obj_max = obj - params.get('sigma', 0.01) * t * dgrad
        enough_descent = objn < obj_max + tol
        if final_step or enough_descent:
            break
        t *= params.get('beta', 0.1)
    else:
        return
    return zn, kan, dobj


def solve(prob, params, z0=None):
    tol = params.get('tol', 1e-10)
    verbose = params.get('verbose', 0)
    lmbda = prob.lmbda
    max_asum = prob.maxAsum
    has_asum = not np.isinf(max_asum)

    if z0 is None:
        n = prob.size
        z = np.zeros(n), 0.0, 0.0
        ka = np.zeros(n)
    else:
        z = z0
        a = z[0]
        assert a.size == prob.size
        idx_sv = a != 0
        ka = prob.k(idx=idx_sv).dot(a[idx_sv]) / lmbda

    recompute_ka = False
    fresh_ka = True
    final_step = False

    for _ in range(params.get('maxSteps', 1000)):
        a, b, c = z
        if final_step:
            break

        if recompute_ka:
            if verbose > 1:
                print('... recompute ka')
            idx_sv = a != 0
            ka = prob.k(idx=idx_sv).dot(a[idx_sv]) / lmbda
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
        if verbose > 0:
            print(f'[{_+1:5d}] f = {obj:10.5g}, r = {res:10.5e}')
        if res < tol:
            if fresh_ka:
                final_step = True
            else:
                recompute_ka = True
                continue

        # try Newton step
        result = descent_step(
            prob, z, ka, dec, obj, g, r,
            final_step=final_step,
            params=params,
        )
        if result is not None:
            z, ka, dobj = result
            if dobj ** 2 < tol:
                recompute_ka = True
            fresh_ka = False
            continue
        else:
            if not fresh_ka:
                recompute_ka = True
                continue

        if final_step:
            break
        result = descent_step(
            prob, z, ka, dec, obj, g, r,
            direction='gradient',
            params=params,
        )
        if result is not None:
            z, ka, _dobj = result
            fresh_ka = False
            continue
        raise RuntimeError('step failed')

    if verbose > 0:
        print('--- solved ---')
    return dict(a=a, b=b, c=c, ka=ka)


def solve_twostep(prob, params_smo, params_newton, verbose=0):
    args = dict(
        x=prob._x.tolist(),
        y=prob._y.tolist(),
        gamma=1.0,
        lmbda=prob.lmbda,
        smoothingParam=prob.smoothingParam,
        maxAsum=prob.maxAsum,
        **params_smo,
    )
    t0 = time.time()
    if isinstance(prob, Classification):
        result = nimsmo.solveClassification(**args, shift=prob.shift)
    if isinstance(prob, Regression):
        result = nimsmo.solveRegression(**args, epsilon=prob.epsilon)
    dt_smo = time.time() - t0
    if verbose > 0:
        print(f't(SMO) = {dt_smo:10.1f}s')
    a = np.array(result['a'])
    b = result['b']
    c = result['c']
    t0 = time.time()
    result = solve(
        prob,
        z0=(a, b, c),
        params=params_newton,
    )
    dt_refine = time.time() - t0
    if verbose > 0:
        print(f't(ref) = {dt_refine:10.1f}s')
    return result
