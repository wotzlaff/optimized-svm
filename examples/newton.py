import numpy as np
from optimized_svm import Regression, rbf_kernel
from optimized_svm.newton import solve_twostep
from sample import data


def main():
    x, y = data(1000, 10, classification=False)

    gamma = 0.2

    params = dict(
        lmbda=1e-3,
        kernel_param=dict(gamma=1.0),
        smoothingParam=0.1,
        shift=1.0,
        epsilon=0.5,
        smo=dict(
            verbose=0,
            tol=0.1,
        ),
        newton=dict(
            verbose=2,
            tol=1e-12,
        ),
        maxfunDegree='2',
    )

    prob = Regression(gamma * x, y, params, kernel_fun=rbf_kernel)
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
