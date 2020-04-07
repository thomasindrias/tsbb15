import numpy as np
#from lk_tracker import lk_tracker
from matplotlib import pyplot as plt
import PIL.Image
from LK_equation import LK_equation
from single_scale_error import LK_errors
import lab2


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


(I, J) = get_cameraman()

plt.figure(1)
plt.imshow(I)


(ds, interpolated_J) = LK_equation(I, J, 15, 1.6, [10, 10])

#plt.imshow(interpolated_J, cmap="gray")
#plt.show()

plt.figure(2)
plt.imshow(ds[:, :, 0, 0])

(err1, err2) = LK_errors(I, J, interpolated_J)

plt.figure(3)
lab2.gopimage(ds[:, :, :, 0])
plt.show()

'''
plt.figure(1)
plt.imshow(ds[:, :, 0, 0], cmap="gray")
plt.figure(2)
plt.imshow(I, cmap="gray")
plt.figure(3)
plt.imshow(J, cmap="gray")
plt.show()
'''
