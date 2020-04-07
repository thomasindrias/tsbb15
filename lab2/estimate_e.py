import numpy as np
from scipy.signal import convolve2d as conv2
from scipy import signal
from matplotlib import pyplot as plt


def estimate_e(Ig, Jg, Jgdx, Jgdy, sigma,  window_size):
    e = np.zeros((Ig.shape[0], Ig.shape[1], 2, 1))

    lp = np.atleast_2d(
        np.exp(-0.5 * np.square(np.arange(-window_size[0]//2, window_size[1]//2 + 1, 1)/sigma)))
    lp = lp / np.sum(lp)

    lp2d = conv2(lp, np.transpose(lp))

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
        e[:, :, 0, 0], lp2d, mode="same")
    e_sum[:, :, 1, 0] = signal.fftconvolve(
        e[:, :, 1, 0], lp2d, mode="same")

    return e_sum
