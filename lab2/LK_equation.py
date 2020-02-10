from regularized_values import regularized_values
from estimate_T import estimate_T
from estimate_e import estimate_e
import numpy as np
from scipy.interpolate import RectBivariateSpline
from matplotlib import pyplot as plt


def LK_equation(I, J, ksize, sigma, w_size):
    (Ig, Jg, Jgdx, Jgdy) = regularized_values(I, J, ksize, sigma)

    #plt.imshow(Jg, cmap="gray")
    # plt.show()

    T = estimate_T(Jgdx, Jgdy, w_size)
    e = estimate_e(Ig, Jg, Jgdx, Jgdy, w_size)

    # d = np.linalg.solve(T[85, 120, :, :], e[85, 120, :, :])

    ds = np.zeros((T.shape[0], T.shape[1], 2, 1))

    interpolator_J = RectBivariateSpline(
        np.arange(Jg.shape[0]), np.arange(Jg.shape[1]), J)

    interpolated_J = np.zeros((Jg.shape[0], Jg.shape[1]))

    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            ds[i, j] = np.linalg.solve(T[i, j, :, :], e[i, j, :, :])
            interpolated_J[i, j] = interpolator_J(np.arange(i + ds[i, j, 0, 0] - 1, i + ds[i, j, 0, 0] + 1, 1),
                                                  np.arange(j + ds[i, j, 1, 0] - 1, j + ds[i, j, 1, 0] + 1, 1))[1, 1]

    return (ds, interpolated_J)
