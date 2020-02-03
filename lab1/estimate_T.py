import numpy as np
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot as plt


def estimate_T(Jgdx, Jgdy, x, y, window_size):

    T = np.zeros((2, 2))

    Jgdx_windowed = Jgdx[x-(window_size[1]//2):(x+(window_size[1]//2)),
                         y - (window_size[0]//2):y + (window_size[0]//2)]

    Jgdy_windowed = Jgdy[x-(window_size[1]//2):(x+(window_size[1]//2)),
                         y - (window_size[0]//2):y + (window_size[0]//2)]

    T[0, 0] = np.sum(Jgdx_windowed*Jgdx_windowed)
    T[0, 1] = np.sum(Jgdx_windowed*Jgdy_windowed)
    T[1, 0] = np.sum(Jgdy_windowed*Jgdx_windowed)
    T[1, 1] = np.sum(Jgdy_windowed*Jgdy_windowed)

    return T


'''
    # If we want to low pass Tensor?
    sigma = 5
    lp = np.atleast_2d(np.exp(-0.5 * (np.arange(-10,11,1)/sigma)**2))
    lp = lp/np.sum(lp)

    Tlp = np.zeros((257,257,3))

    Tlp[:,:,0] = conv2(conv2(T[:,:,0], lp, mode='same'), lp.T, mode='same')
    Tlp[:,:,1] = conv2(conv2(T[:,:,1], lp, mode='same'), lp.T, mode='same')
    Tlp[:,:,2] = conv2(conv2(T[:,:,2], lp, mode='same'), lp.T, mode='same')
'''
