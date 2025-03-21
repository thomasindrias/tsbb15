import numpy as np
# from lk_tracker import lk_tracker
from matplotlib import pyplot as plt
import PIL.Image
from single_scale_error import LK_errors
import lab2
from LK_equation import LK_equation


def load_image_grayscale(path):
    "Load a grayscale image by path"
    return np.asarray(PIL.Image.open(path).convert('L'))


I = load_image_grayscale('forwardL/forwardL0.png')
J = load_image_grayscale('forwardL/forwardL1.png')

ds, interpolated_J = LK_equation(I, J, 11, 1.0, [10, 10], True)

plt.imshow(interpolated_J, cmap="gray")
plt.show()

(err1, err2) = LK_errors(I, J, interpolated_J)

lab2.gopimage(ds[:, :, :, 0])
plt.show()
