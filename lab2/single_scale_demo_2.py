import numpy as np
# from lk_tracker import lk_tracker
from matplotlib import pyplot as plt
import PIL.Image
from scipy import ndimage
from regularized_values import regularized_values
from estimate_T import estimate_T
from estimate_e import estimate_e
from single_scale_error import single_scale_error
from scipy.interpolate import RectBivariateSpline
import lab2
import scipy


def load_image_grayscale(path):
    "Load a grayscale image by path"
    return np.asarray(PIL.Image.open(path).convert('L'))


def get_cameraman():
    "Return I, J and true (col, row) displacement"
    n = 10  # Border crop
    img = load_image_grayscale('cameraman.tif')
    I = img[n:-n, n:-n]
    x, y = 1, -2
    J = img[n-y:-n-y, n-x:-n-x]
    assert I.shape == J.shape
    return I, J


I = load_image_grayscale('forwardL/forwardL0.png')
J = load_image_grayscale('forwardL/forwardL1.png')

(I, J) = get_cameraman()

(Ig, Jg, Jgdx, Jgdy) = regularized_values(I, J, 11, 2.0)

T = estimate_T(Jgdx, Jgdy, [40, 70])
e = estimate_e(Ig, Jg, Jgdx, Jgdy, [40, 70])

# d = np.linalg.solve(T[85, 120, :, :], e[85, 120, :, :])


ds = np.zeros((T.shape[0], T.shape[1], 2, 1))

interpolator_J = RectBivariateSpline(
    np.arange(Jg.shape[0]), np.arange(Jg.shape[1]), J)

interpolated_J = np.zeros((Jg.shape[0], Jg.shape[1]))

for i in range(T.shape[0]):
    for j in range(T.shape[1]):
        ds[i, j] = np.linalg.solve(T[i, j, :, :], e[i, j, :, :])
        interpolated_J[i, j] = interpolator_J(np.arange(i + ds[i, j, 0, 0] - 1, i + ds[i, j, 0, 0] + 1, 1),
                                              np.arange(j + ds[i, j, 1, 0] - 1, j + ds[i, j, 1, 0] + 1, 1))[1, 1]


#plt.imshow(interpolated_J, cmap="gray")
# plt.show()

err1 = single_scale_error(I, J)
err2 = single_scale_error(I, interpolated_J)

lab2.gopimage(ds[:, :, :, 0])
plt.show()

print("||J(x) - I(x)||", err1)
print("||J(x+v) - I(x)||", err2)
