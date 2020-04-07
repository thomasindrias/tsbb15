import numpy as np
from matplotlib import pyplot as plt
import PIL.Image
from single_scale_error import LK_errors
from LK_equation import LK_equation
import lab2

def multiscale():
    d_tot = np.zeros((J.shape[0], J.shape[1], 2, 1))
    Jn = J

    n_scales = 7

    for n in range(n_scales, 2, -1):
        sc = 2 ** (n - 1)
        dn, Cn = LK_equation(I, Jn, sc * 2.0, sc * 0.1, [40, 40], remove_outliers=True)
        d_tot += dn

        Jn = Cn

        lab2.gopimage(d_tot[:, :, :, 0])
        plt.show()

        (err1, err2) = LK_errors(I, J, Jn)

    return d_tot[:,:,:,0]


def load_image_grayscale(path):
    "Load a grayscale image by path"
    return np.asarray(PIL.Image.open(path).convert('L'))

I = load_image_grayscale('SCcar4/SCcar4_00070.bmp')
J = load_image_grayscale('SCcar4/SCcar4_00071.bmp')


d = np.zeros((J.shape[0], J.shape[1]))
