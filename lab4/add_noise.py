import numpy as np


def add_noise(im, snr):
    gaussian = np.random.normal(0, 10, (im.shape[0], im.shape[1]))

    current_snr = np.mean(im) / np.std(gaussian)

    gaussian *= (current_snr / snr)

    return im + gaussian
