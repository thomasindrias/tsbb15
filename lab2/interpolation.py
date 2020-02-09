import numpy as np
from scipy.signal import convolve2d as conv2
from scipy.interpolate import RectBivariateSpline
from matplotlib import pyplot as plt


def init_interpolation(X):
    interpolator = RectBivariateSpline(
        np.arange(X.shape[0]), np.arange(X.shape[1]), X)
    return interpolator


def interpolate(interpolator, X, dx, dy):
    interpolated = interpolator(np.arange(dy, dy+X.shape[0], 1),
                                np.arange(dx, dx+X.shape[1], 1))
    return interpolated
