import numpy as np
#from lk_tracker import lk_tracker
from matplotlib import pyplot as plt
import PIL.Image
from scipy import ndimage
from regularized_values import regularized_values
from estimate_T import estimate_T
from estimate_e import estimate_e
#import lab1
import scipy


def load_image_grayscale(path):
    "Load a grayscale image by path"
    return np.asarray(PIL.Image.open(path).convert('L'))


I = load_image_grayscale('forwardL/forwardL0.png')
J = load_image_grayscale('forwardL/forwardL1.png')

(Ig, Jg, Jgdx, Jgdy) = regularized_values(I, J, 11, 2.0)

T = estimate_T(Jgdx, Jgdy, [40, 70])
e = estimate_e(Ig, Jg, Jgdx, Jgdy, [40, 70])

#d = np.linalg.solve(T[85, 120, :, :], e[85, 120, :, :])


ds = np.zeros((T.shape[0], T.shape[1], 2, 1))
for i in range(T.shape[0]):
    for j in range(T.shape[1]):
        ds[i, j] = np.linalg.solve(T[i, j, :, :], e[i, j, :, :])

'''
plt.figure(1)
plt.imshow(ds[:, :, 0, 0], cmap="gray")
plt.figure(2)
plt.imshow(I, cmap="gray")
plt.figure(3)
plt.imshow(J, cmap="gray")
plt.show()
'''
