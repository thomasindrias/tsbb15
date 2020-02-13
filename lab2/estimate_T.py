import numpy as np
from scipy.signal import convolve2d as conv2
from scipy import signal
from matplotlib import pyplot as plt


def estimate_T(Jgdx, Jgdy, sigma, window_size):

    T = np.zeros((Jgdx.shape[0], Jgdx.shape[1], 2, 2))

    lp = np.atleast_2d(
        np.exp(-0.5 * np.square(np.arange(-window_size[0]//2, window_size[1]//2 + 1, 1)/sigma)))
    lp = lp / np.sum(lp)

    lp2d = conv2(lp, np.transpose(lp))

    T[:, :, 0, 0] = Jgdx * Jgdx
    T[:, :, 0, 1] = Jgdx * Jgdy
    T[:, :, 1, 0] = T[:, :, 0, 1]
    T[:, :, 1, 1] = Jgdy * Jgdy

    sum_T = np.zeros((Jgdx.shape[0], Jgdx.shape[1], 2, 2))

    sum_T[:, :, 0, 0] = signal.fftconvolve(
        T[:, :, 0, 0], lp2d, mode="same")
    sum_T[:, :, 0, 1] = signal.fftconvolve(
        T[:, :, 0, 1], lp2d, mode="same")
    sum_T[:, :, 1, 0] = sum_T[:, :, 0, 1]
    sum_T[:, :, 1, 1] = signal.fftconvolve(
        T[:, :, 1, 1], lp2d, mode="same")

    return sum_T
