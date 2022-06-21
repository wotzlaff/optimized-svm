from functools import partial
import numpy as np


def smooth_max_p2(t, s):
    return np.piecewise(
        t, [t <= -s, t >= s],
        [0.0, lambda t: t, lambda t: 0.25 / s * (t + s)**2]
    )


def dsmooth_max_p2(t, s):
    return np.piecewise(
        t, [t <= -s, t >= s], [0.0, 1.0, lambda t: 0.5 / s * (t + s)]
    )


def ddsmooth_max_p2(t, s):
    return np.piecewise(t, [t <= -s, t >= s], [0.0, 0.0, 0.5 / s])


def smooth_max_p3(t, s):
    return np.piecewise(
        t, [t <= -s, t >= s, (t <= 0.0) & (t > -s)],
        [
            0.0,
            lambda t: t,
            lambda t: (s + t) ** 3 / (6 * s**2),
            lambda t: t + (s - t) ** 3 / (6 * s**2)
        ]
    )


def dsmooth_max_p3(t, s):
    return np.piecewise(
        t, [t <= -s, t >= s, (t <= 0.0) & (t > -s)],
        [
            0.0,
            1.0,
            lambda t: (s + t) ** 2 / (2 * s**2),
            lambda t: 1.0 - (s - t) ** 2 / (2 * s**2)
        ]
    )


def ddsmooth_max_p3(t, s):
    return np.piecewise(
        t, [t <= -s, t >= s, (t <= 0.0) & (t > -s)],
        [
            0.0,
            0.0,
            lambda t: (s + t) / s**2,
            lambda t: (s - t) / s**2
        ]
    )


def smooth_max_p6(t, s):
    return np.piecewise(t, [t <= -s, t >= s],[
        0.0,
        lambda t: t,
        lambda t: s * (5 + t/s * (16 + t/s * (15 + t/s * t/s * (-5 + t/s * t/s)))) / 32
    ])


def dsmooth_max_p6(t, s):
    return np.piecewise(t, [t <= -s, t >= s], [
        0.0,
        1.0,
        lambda t: (8 + t/s * (15 + t/s * t/s * (-10 + 3 * t/s * t/s))) / 16
    ])


def ddsmooth_max_p6(t, s):
    return np.piecewise(t, [t <= -s, t >= s], [
        0.0, 0.0,
        lambda t: (15 + t/s * t/s * (-30 + 15 * t/s * t/s)) / (16 * s)
    ])


def maxfun(smoothingParam, deg='6'):
    return [
        partial(fun, s=smoothingParam)
        for fun in {
            '2': [smooth_max_p2, dsmooth_max_p2, ddsmooth_max_p2],
            '3': [smooth_max_p3, dsmooth_max_p3, ddsmooth_max_p3],
            '6': [smooth_max_p6, dsmooth_max_p6, ddsmooth_max_p6],
        }[deg]
    ]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    f, axs = plt.subplots(3, 3)
    for j, deg in enumerate(['2', '3', '6']):
        mf = maxfun(0.2, deg)
        for i, f in enumerate(mf):
            x = np.linspace(-0.5, 0.5, 101)
            y = f(x)
            axs[i, j].plot(x, y, 'k-')
    plt.show()
