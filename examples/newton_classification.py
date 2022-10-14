import numpy as np
from optimized_svm import Classification, rbf_kernel
from optimized_svm.newton import solve_twostep
from sample import data


def main():
    x, y = data(1000, 10, classification=True)

    gamma = 0.2

    params = dict(
        lmbda=1e-2,
        smoothingParam=0.1,
        maxfunDegree='2',
    )
    params_smo = dict(
        verbose=100,
        tol=1e-10,
        shrinkingPeriod=500,
        logObjective=True,
    )
    params_newton = dict(
        verbose=2,
        tol=1e-12,
    )

    prob = Classification(
        gamma * x, y,
        kernel_fun=rbf_kernel,
        kernel_param=dict(gamma=1.0),
        w=np.random.rand(y.size),
        **params,
    )
    result = solve_twostep(
        prob,
        params_smo,
        params_newton,
    )
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
