from regularized_values import regularized_values
from estimate_T import estimate_T
from estimate_e import estimate_e
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.signal import medfilt2d


def LK_equation(I, J, ksize, sigma, w_size, remove_outliers=False):
    (Ig, Jg, Jgdx, Jgdy) = regularized_values(I, J, ksize, sigma)

    T = estimate_T(Jgdx, Jgdy, sigma, w_size)
    e = estimate_e(Ig, Jg, Jgdx, Jgdy, sigma, w_size)

    ds = np.linalg.solve(T, e)

    if remove_outliers:
        ds[:, :, 0, 0] = medfilt2d(ds[:, :, 0, 0], 5)
        ds[:, :, 1, 0] = medfilt2d(ds[:, :, 1, 0], 5)

    interpolator_J = RectBivariateSpline(
        np.arange(Jg.shape[0]), np.arange(Jg.shape[1]), J)

    interpolated_J = np.zeros((Jg.shape[0], Jg.shape[1]))

    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            interpolated_J[i, j] = interpolator_J(np.arange(i + ds[i, j, 1, 0] - 1, i + ds[i, j, 1, 0] + 1, 1),
                                                  np.arange(j + ds[i, j, 0, 0] - 1, j + ds[i, j, 0, 0] + 1, 1))[1, 1]

    return (ds, interpolated_J)
