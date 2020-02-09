import numpy as np
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot as plt


def regularized_values(I, J, ksize, sigma):
    lp = np.atleast_2d(
        np.exp(-0.5 * np.square(np.arange(-ksize//2, ksize//2 + 1, 1)/sigma)))
    lp = lp / np.sum(lp)

    lp2d = conv2(lp, np.transpose(lp))

    Ig = conv2(I, lp2d, mode='same')
    Jg = conv2(J, lp2d, mode='same')

    df = np.atleast_2d(-1.0/np.square(sigma) *
                       np.arange(-ksize//2, ksize//2 + 1, 1) * lp)

    Jgdx = conv2(conv2(J, df, mode='same'), lp.T, mode='same')
    Jgdy = conv2(conv2(J, lp, mode='same'), df.T, mode='same')

    return (Ig, Jg, Jgdx, Jgdy)
