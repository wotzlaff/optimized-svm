import numpy as np
from optimized_svm import OptimizedSVM, SimpleSVM, rbf_kernel
from sample import data


def main():
    verbose = 1
    classification = False
    n = 500
    x, y = data(
        n, nft=5,
        relevant_ratio=0.2,
        classification=classification,
    )
    # x[:, 0] *= 10
    kind = 'classification' if classification else 'regression'
    params = dict(
        lmbda=1.0,
        smoothingParam=0.5,
        maxfunDegree='2',
        # maxAsum=0.01 * n * 0.8,
    )
    scaling = 1.0
    if classification:
        params['shift'] = 1.0
    else:
        params['epsilon'] = 0.0
    params_smo = dict(
        verbose=0,
        shrinkingPeriod=1000,
        tol=1e-3,
    )
    params_newton = dict(
        verbose=0,
        tol=1e-12,
    )
    svm0 = SimpleSVM(
        kind=kind,
        params=params,
        scaling=scaling,
        kernel_fun=rbf_kernel,
        kernel_param=dict(gamma=1.0),
        verbose=verbose,
        params_smo=params_smo,
        params_newton=params_newton,
    )
    svm0.fit(x[:n//2], y[:n//2])
    yest0 = svm0.predict(x)

    svm = OptimizedSVM(
        kind=kind,
        params=params,
        kernel_fun=rbf_kernel,
        kernel_param=dict(
            gamma=1.0,
            blocks='all',
            # blocks=[[i] for i in range(x.shape[1])],
            # blocks=[[0], [1]],
        ),
        params_hpo=dict(
            scaling0=scaling,
            lmbda0=params['lmbda'],
            ftol=1e-6,
            gtol=1e-6,
            iprint=1,
        ),
        validation_params=dict(
            kind=kind,
        ),
        # debug=True,
        verbose=verbose,
        params_smo=params_smo,
        params_newton=params_newton,
    )
    svm.fit(x[:n//2], y[:n//2])
    yest = svm.predict(x)

    import matplotlib.pyplot as plt
    if kind == 'classification':
        plt.plot(
            y[:n//2] * 0.9, y[:n//2] * yest[:n//2],
            'kx', label='after(train)',
        )
        plt.plot(
            y[n//2:], y[n//2:] * yest[n//2:],
            'g+', label='after(val)',
        )
        plt.plot(
            y[:n//2] * 0.7, y[:n//2] * yest0[:n//2],
            'bx', label='intial(train)',
        )
        plt.plot(
            y[n//2:] * 0.8, y[n//2:] * yest0[n//2:],
            'r+', label='initial(val)',
        )
        plt.legend()

        mc0 = (y[n//2:] * yest0[n//2:] <= 0.0).mean()
        mc = (y[n//2:] * yest[n//2:] <= 0.0).mean()
        print(f'MC: {mc0} -> {mc}')

        plt.axhline(1.0)
        plt.axhline(0.0)
    elif kind == 'regression':
        plt.plot(y[:n//2], yest0[:n//2], 'bo', label='intial(train)')
        plt.plot(y[n//2:], yest0[n//2:], 'ro', label='initial(val)')
        plt.plot(y[:n//2], yest[:n//2], 'kx', label='after(train)')
        plt.plot(y[n//2:], yest[n//2:], 'g+', label='after(val)')
        vmin = min(y.min(),  yest.min())
        vmax = max(y.max(),  yest.max())
        plt.plot([vmin, vmax], [vmin, vmax], '--')
        plt.legend()

        mae0 = np.abs(y[n//2:] - yest0[n//2:]).mean()
        mae = np.abs(y[n//2:] - yest[n//2:]).mean()
        print(f'MAE: {mae0} -> {mae}')
        maem = np.abs(y[n//2:] - np.median(y[n//2:])).mean()
        print(f'rMAE: {mae0/maem} -> {mae/maem}')
    plt.show()


if __name__ == '__main__':
    main()
