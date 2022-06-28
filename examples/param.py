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
    params = dict(
        lmbda=1.0,
        scaling=1.0,
        kernel_param=dict(gamma=1.0),
        smoothingParam=0.02,
        epsilon=1e-6,
        shift=1.0,
        maxfunDegree='2',
        smo=dict(
            # verbose=1000,
            tol=0.1,
        ),
        newton=dict(
            # verbose=2,
            tol=1e-12,
        ),
        # maxAsum=0.01 * n * 0.8,
        kind='classification' if classification else 'regression',
    )
    svm0 = SimpleSVM(params, kernel_fun=rbf_kernel, verbose=verbose)
    svm0.fit(x[:n//2], y[:n//2])
    yest0 = svm0.predict(x)

    svm = OptimizedSVM(
        params=params,
        validation_params=dict(
            kind='classification' if classification else 'regression',
            # kind='regression',
            shift=1.0,
        ),
        kernel_fun=rbf_kernel,
        blocks=[[i] for i in range(x.shape[1])],
        # blocks=[[0], [1]],
        # debug=True,
        verbose=verbose,
    )
    svm.fit(x[:n//2], y[:n//2])
    yest = svm.predict(x)

    import matplotlib.pyplot as plt
    if svm._params['kind'] == 'classification':
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
    elif svm._params['kind'] == 'regression':
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
