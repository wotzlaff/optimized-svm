import numpy as np
from optimized_svm import OptimizedSVM, rbf_kernel
from sample import data
import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.use('TKAgg')


def main():
    n, nft = 2000, 2
    x, y = data(n, nft, scale=4, classification=False)
    y = np.round((y / nft) ** 2 * 25)

    params = dict(
        lmbda=0.01,
    )
    params_smo = dict()
    params_newton = dict(
        # verbose=2,
        tol=1e-12,
    )

    scaling = 'auto'
    kind = 'poisson'
    svm = OptimizedSVM(
        kind=kind,
        params=params,
        kernel_fun=rbf_kernel,
        kernel_param=dict(gamma=1.0),
        params_hpo=dict(
            normalize_objective=True,
            kind=kind,
            scaling0=scaling,
            lmbda0=params['lmbda'],
            iprint=1,
            lmbda_bounds=(1e-6, 1e3),
            scaling_bounds=(1e-6, 1e3),
        ),
        # debug=True,
        # verbose=3,
        params_smo=params_smo,
        params_newton=params_newton,
    )
    svm.fit(x[:n//2], y[:n//2])
    yest = svm.predict(x)
    
    plt.scatter(x[:n//2, 0], x[:n//2, 1], c='k', s=y[:n//2])
    plt.scatter(x[n//2:, 0], x[n//2:, 1], c='r', s=yest[n//2:])
    plt.show()


if __name__ == '__main__':
    main()
