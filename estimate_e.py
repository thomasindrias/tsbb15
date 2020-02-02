import numpy as np
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot as plt


def estimate_e(Ig, Jg, Jgdx, Jgdy, x, y, window_size):

    e = np.zeros((window_size[1], window_size[0], 2))

    Jgdx_windowed = Jgdx[x-(window_size[1]//2):(x+(window_size[1]//2)),
                         y - (window_size[0]//2):y + (window_size[0]//2)]

    Jgdy_windowed = Jgdy[x-(window_size[1]//2):(x+(window_size[1]//2)),
                         y - (window_size[0]//2):y + (window_size[0]//2)]

    Ig_windowed = Ig[x-(window_size[1]//2):(x+(window_size[1]//2)),
                     y - (window_size[0]//2):y + (window_size[0]//2)]

    Jg_windowed = Jg[x-(window_size[1]//2):(x+(window_size[1]//2)),
                     y - (window_size[0]//2):y + (window_size[0]//2)]

    diff = Ig_windowed - Jg_windowed

    e[:, :, 0] = diff * Jgdx_windowed
    e[:, :, 1] = diff * Jgdy_windowed

    return e


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
