import numpy as np
from scipy import signal


def get_HL(L):
    H11 = np.matrix([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
    H12 = np.matrix([[0.5, 0, -0.5], [0, 0, 0], [-0.5, 0, 0.5]])
    H22 = np.matrix([[0, 1, 0], [0, -2, 0], [0, 1, 0]])

    HL = np.zeros((L.shape[0], L.shape[1], 2, 2))

    HL[:, :, 0, 0] = signal.convolve2d(L, H11, mode="same")
    HL[:, :, 0, 1] = signal.convolve2d(L, H12, mode="same")
    HL[:, :, 1, 0] = HL[:, :, 0, 1]
    HL[:, :, 1, 1] = signal.convolve2d(L, H22, mode="same")

    return HL
