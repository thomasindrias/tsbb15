import regularized_values
import estimate_T
import estimate_e
from harris import orientation_tensor, harris
from interpolation import init_interpolation
from interpolation import interpolate
import numpy as np
import scipy
from scipy import ndimage
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline
import PIL.Image
import lab1

I = lab1.load_lab_image('cornertest.png')

#plt.figure(1)
#plt.imshow(Ig)
#plt.figure(2)
#plt.imshow(I)
#plt.show()

T_field = orientation_tensor(I, 18, 0.5, 18, 1.5)

#plt.imshow(T_field[:,:,1,1], cmap="gray")
#print(T_field[:,:,1,1].shape)
#print(I.shape)
#plt.show()

harris_response = harris(T_field, 0.05)

print(np.max(harris_response))

harris_thresholded1 = harris_response > 10000
#harris_thresholded2 = harris_response > -500
#harris_thresholded = harris_thresholded1 * harris_thresholded2

#harris_thresholded = (harris_response > -1000) & (harris_response < 1000)
#harris_thresholded = harris_thresholded * harris_response

#max_img = ndimage.filters.maximum_filter(harris_thresholded, size=3)
#[row, col] = np.nonzero(harris_thresholded == max_img)
#print(row)
#print(col)

plt.imshow(harris_thresholded1, cmap="gray")
plt.show()

#plt.figure(1)
#plt.imshow(T_field[:, :, 0, 0], cmap="gray")
#plt.figure(2)
#plt.imshow(T_field[:, :, 1, 1], cmap="gray")
#plt.show()
