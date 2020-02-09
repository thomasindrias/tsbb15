import numpy as np
from scipy.signal import convolve2d as conv2
from scipy import signal
from matplotlib import pyplot as plt


def estimate_e(Ig, Jg, Jgdx, Jgdy, window_size):
    e = np.zeros((Ig.shape[0], Ig.shape[1], 2, 1))

    sum_filter = np.ones((window_size))

    '''
        sum_Ig = signal.fftconvolve(Ig, sum_filter, mode="same")
        sum_Jg = signal.fftconvolve(Jg, sum_filter, mode="same")
        sum_Jgdx = signal.fftconvolve(Jgdx, sum_filter, mode="same")
        sum_Jgdy = signal.fftconvolve(Jgdy, sum_filter, mode="same")
    '''

    diff = Ig - Jg

    e[:, :, 0, 0] = diff * Jgdx
    e[:, :, 1, 0] = diff * Jgdy

    e_sum = np.zeros((Ig.shape[0], Ig.shape[1], 2, 1))

    e_sum[:, :, 0, 0] = signal.fftconvolve(
        e[:, :, 0, 0], sum_filter, mode="same")
    e_sum[:, :, 1, 0] = signal.fftconvolve(
        e[:, :, 1, 0], sum_filter, mode="same")

    return e_sum
