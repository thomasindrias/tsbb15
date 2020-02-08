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

def rgb_to_grayscale(I):
    G = 0.299*I[:, :, 0] + 0.587*I[:, :, 1] + 0.114*I[:, :, 2]
    return G

def load_lab_image(filename):
    """Load a grayscale image by filename from the CVL image directory

    Example:
    >>> img = load_lab_image('cornertest.png')
    """

    return np.asarray(PIL.Image.open(filename).convert('L'))

def get_cameraman():
    "Return I, J and true (col, row) displacement"
    n = 10 # Border crop
    img = load_lab_image('cameraman.tif')
    I = img[n:-n, n:-n]
    x, y = 1, -2
    J = img[n-y:-n-y, n-x:-n-x]
    assert I.shape == J.shape
    return I, J, (x, y)

I, J, (x,y) = get_cameraman()

d_true = [[x], [y]]
w_size = [70, 40]

(Ig, Jg, Jgdx, Jgdy) = regularized_values.regularized_values(I, J, 11, 2.0)

#plt.imshow(Jgdx)
#plt.show()

T = estimate_T.estimate_T(Jgdx, Jgdy, 52, 114, w_size)

plt.imshow(T)
plt.show()
e = estimate_e.estimate_e(Ig, Jg, Jgdx, Jgdy, 52, 114, w_size)

# Find an appropiate value, d
d_tot = np.zeros((2, 1))
#print(d_tot)

# d = T^-1 * e
d = np.linalg.solve(T, e)
d_diff = d_tot - np.asarray(d_true)

counter = 0

interpolator_Jg = RectBivariateSpline(
    np.arange(Jg.shape[0]), np.arange(Jg.shape[1]), Jg)

interpolator_Jgdx = RectBivariateSpline(
    np.arange(Jgdx.shape[0]), np.arange(Jgdx.shape[1]), Jgdx)

interpolator_Jgdy = RectBivariateSpline(
    np.arange(Jgdy.shape[0]), np.arange(Jgdy.shape[1]), Jgdy)


while np.linalg.norm(d) > 0.0001 and counter < 10:
    d_tot = d_tot + d

    dx = d_tot[0][0]
    dy = d_tot[1][0]

    Jg_interpolated = interpolate(interpolator_Jg, Jg, dx, dy)

    Jgdx_interpolated = interpolate(interpolator_Jgdx, Jgdx, dx, dy)

    Jgdy_interpolated = interpolate(interpolator_Jgdy, Jgdy, dx, dy)

    T = estimate_T.estimate_T(Jgdx_interpolated, Jgdy_interpolated, 52, 114, w_size)
    e = estimate_e.estimate_e(Ig, Jg_interpolated, Jgdx_interpolated, Jgdy_interpolated, 52, 114, w_size)

    d = np.linalg.solve(T, e)
    d_diff = d_tot - np.asarray(d_true)

    #print("COUNTER", counter)
    #print("D_TOT", d_tot)

    counter = counter + 1

#plt.figure(1)
#plt.imshow(Ig)
#plt.figure(2)
#plt.imshow(Jg_interpolated)
#plt.show()

T_field = orientation_tensor(I, 17, 2.0, 17, 2.0)

#plt.imshow(T_field[:,:,1,1], cmap="gray")
#print(T_field[:,:,1,1].shape)
#print(I.shape)
#plt.show()

harris_response = harris(T_field, 0.05)

#harris_thresholded1 = harris_response < 500
#harris_thresholded2 = harris_response > -500
#harris_thresholded = harris_thresholded1 * harris_thresholded2

harris_thresholded = (harris_response < -15000) | (harris_response > 15000)
#harris_thresholded = harris_thresholded * harris_response

max_img = ndimage.filters.maximum_filter(harris_thresholded, size=1)
[row, col] = np.nonzero(harris_thresholded == max_img)
print(row)
print(col)

plt.imshow(max_img, cmap="gray")
plt.show()
