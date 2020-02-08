import estimate_T
import estimate_e
from interpolation import interpolate
import numpy as np
from scipy.interpolate import RectBivariateSpline

def lk_tracker(Ig, Jg, Jgdx, Jgdy, tracking_point):
    tx = tracking_point[0]
    ty = tracking_point[1]

    w_size = [100, 100]

    T = estimate_T.estimate_T(Jgdx, Jgdy, tx, ty, w_size)
    e = estimate_e.estimate_e(Ig, Jg, Jgdx, Jgdy, tx, ty, w_size)

    # Find an appropiate value, d
    d_tot = np.zeros((2, 1))
    #print(d_tot)

    # d = T^-1 * e
    d = np.linalg.solve(T, e)

    counter = 0

    interpolator_Jg = RectBivariateSpline(
        np.arange(Jg.shape[0]), np.arange(Jg.shape[1]), Jg)

    interpolator_Jgdx = RectBivariateSpline(
        np.arange(Jgdx.shape[0]), np.arange(Jgdx.shape[1]), Jgdx)

    interpolator_Jgdy = RectBivariateSpline(
        np.arange(Jgdy.shape[0]), np.arange(Jgdy.shape[1]), Jgdy)


    while np.linalg.norm(d) > 0.0001 and counter < 100:
        d_tot = d_tot + d

        dx = d_tot[0][0]
        dy = d_tot[1][0]

        Jg_interpolated = interpolate(interpolator_Jg, Jg, dx, dy)

        Jgdx_interpolated = interpolate(interpolator_Jgdx, Jgdx, dx, dy)

        Jgdy_interpolated = interpolate(interpolator_Jgdy, Jgdy, dx, dy)

        T = estimate_T.estimate_T(Jgdx_interpolated, Jgdy_interpolated, tx, ty, w_size)
        e = estimate_e.estimate_e(Ig, Jg_interpolated, Jgdx_interpolated, Jgdy_interpolated, tx, ty, w_size)

        d = np.linalg.solve(T, e)

        #print("COUNTER", counter)
        #print("D_TOT", d_tot)

        counter = counter + 1

    return (dx, dy)
