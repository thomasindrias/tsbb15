import numpy as np
from scipy.signal import convolve2d as conv2
from scipy import signal
from matplotlib import pyplot as plt


def estimate_T(img, gradksize, gradsigma, ksize, sigma):
    lp = np.atleast_2d(
        np.exp(-0.5 * np.square(np.arange(-ksize // 2, ksize // 2 + 1, 1) / sigma))
    )
    lp = lp / np.sum(lp)

    df = np.atleast_2d(
        -1.0 / np.square(sigma) * np.arange(-ksize // 2, ksize // 2 + 1, 1) * lp
    )

    img_dx = conv2(conv2(img, df, mode="same"), lp.T, mode="same")
    img_dy = conv2(conv2(img, lp, mode="same"), df.T, mode="same")

    T_field = np.zeros((img.shape[0], img.shape[1], 2, 2))

    T_field[:, :, 0, 0] = img_dx * img_dx
    T_field[:, :, 0, 1] = img_dx * img_dy
    T_field[:, :, 1, 0] = img_dy * img_dx
    T_field[:, :, 1, 1] = img_dy * img_dy

    lp = np.atleast_2d(
        np.exp(
            -0.5 * (np.arange(-gradksize // 2, gradksize // 2 + 1, 1) / gradsigma) ** 2
        )
    )
    lp = lp / np.sum(lp)

    Tlp = np.zeros((img.shape[0], img.shape[1], 2, 2))

    Tlp[:, :, 0, 0] = conv2(
        conv2(T_field[:, :, 0, 0], lp, mode="same"), lp.T, mode="same"
    )
    Tlp[:, :, 0, 1] = conv2(
        conv2(T_field[:, :, 0, 1], lp, mode="same"), lp.T, mode="same"
    )
    Tlp[:, :, 1, 0] = conv2(
        conv2(T_field[:, :, 1, 0], lp, mode="same"), lp.T, mode="same"
    )
    Tlp[:, :, 1, 1] = conv2(
        conv2(T_field[:, :, 1, 1], lp, mode="same"), lp.T, mode="same"
    )

    return Tlp
