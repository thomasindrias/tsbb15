import numpy as np
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot as plt


def regularized_values(img, ksize, sigma):
    lp = np.atleast_2d(
        np.exp(-0.5 * np.square(np.arange(-ksize // 2, ksize // 2 + 1, 1) / sigma))
    )
    lp = lp / np.sum(lp)
    df = np.atleast_2d(
        -1.0 / np.square(sigma) * np.arange(-ksize //
                                            2, ksize // 2 + 1, 1) * lp
    )

    img_dx = conv2(conv2(img, df, mode="same"), lp.T, mode="same")
    img_dy = conv2(conv2(img, lp, mode="same"), df.T, mode="same")

    return (img_dx, img_dy)
