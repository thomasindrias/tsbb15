import lab3
from part4 import get_corr
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

plt.set_cmap("gray")

[im1, im2] = lab3.load_stereo_pair()

(corr1, corr2) = get_corr(im1, im2)
