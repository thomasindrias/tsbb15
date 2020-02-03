import numpy as np
from scipy.signal import convolve2d as conv2
from scipy.interpolate import RectBivariateSpline
from matplotlib import pyplot as plt

img = plt.imread('image.png')
img = 0.299*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]

plt.figure(1)
plt.imshow(img[100:102, 100:102], cmap="gray")
# plt.show()

imgsc = RectBivariateSpline(
    np.arange(img.shape[0]), np.arange(img.shape[1]), img)


interpolated = imgsc(np.arange(100, 102, 0.5),
                     np.arange(100, 102, 0.5))

print(interpolated.shape)
plt.figure(2)
plt.imshow(interpolated, cmap="gray")
plt.show()
