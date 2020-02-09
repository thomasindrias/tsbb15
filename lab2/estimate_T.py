import numpy as np
from scipy.signal import convolve2d as conv2
from scipy import signal
from matplotlib import pyplot as plt


def estimate_T(Jgdx, Jgdy, window_size):

    T = np.zeros((Jgdx.shape[0], Jgdx.shape[1], 2, 2))

    sum_filter = np.ones((window_size))

    T[:, :, 0, 0] = Jgdx * Jgdx
    T[:, :, 0, 1] = Jgdx * Jgdy
    T[:, :, 1, 0] = T[:, :, 0, 1]
    T[:, :, 1, 1] = Jgdy * Jgdy

    sum_T = np.zeros((Jgdx.shape[0], Jgdx.shape[1], 2, 2))

    sum_T[:, :, 0, 0] = signal.fftconvolve(
        T[:, :, 0, 0], sum_filter, mode="same")
    sum_T[:, :, 0, 1] = signal.fftconvolve(
        T[:, :, 0, 1], sum_filter, mode="same")
    sum_T[:, :, 1, 0] = sum_T[:, :, 0, 1]
    sum_T[:, :, 1, 1] = signal.fftconvolve(
        T[:, :, 1, 1], sum_filter, mode="same")

    return sum_T
