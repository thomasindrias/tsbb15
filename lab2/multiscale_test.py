import numpy as np
# from lk_tracker import lk_tracker
from matplotlib import pyplot as plt
import PIL.Image
from scipy import ndimage
from single_scale_error import single_scale_error
from scipy.interpolate import RectBivariateSpline
import lab2
import scipy
from LK_equation import LK_equation
from scipy.signal import medfilt2d


def load_image_grayscale(path):
    "Load a grayscale image by path"
    return np.asarray(PIL.Image.open(path).convert('L'))


I = load_image_grayscale('SCcar4/SCcar4_00070.bmp')
J = load_image_grayscale('SCcar4/SCcar4_00071.bmp')

#plt.imshow(interpolated_J, cmap="gray")
# plt.show()


d_tot = np.zeros((J.shape[0], J.shape[1], 2, 1))
Jn = J

n_scales = 4

for n in range(n_scales, 1, -1):
    sc = 2 ** (n - 1)
    dn, Cn = LK_equation(I, Jn, sc * 11, sc * 2.0, [40, 70])
    d_tot += dn
    #Jn = medfilt2d(Cn, 3)
    Jn = Cn
    # plt.figure(n)
    #plt.imshow(Jn, cmap="gray")

    err1 = single_scale_error(I, J)
    err2 = single_scale_error(I, Jn)

# plt.show()
