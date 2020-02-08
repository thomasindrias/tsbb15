import numpy as np
import PIL.Image
import scipy
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

import estimate_e
import estimate_T
import regularized_values
from harris import harris, orientation_tensor
from interpolation import init_interpolation, interpolate
import lab1


I = lab1.load_lab_image('cornertest.png')

#plt.figure(1)
#plt.imshow(Ig)
#plt.figure(2)
#plt.imshow(I)
#plt.show()

T_field = orientation_tensor(I, 11, 2.0, 11, 1.5)

#plt.imshow(T_field[:,:,1,1], cmap="gray")
#print(T_field[:,:,1,1].shape)
#print(I.shape)
#plt.show()

harris_response = harris(T_field, 0.05)

harris_thresholded = harris_response > 100000
harris_thresholded = harris_thresholded * 1

[ht1, ht2] = np.nonzero(harris_thresholded)

domain = np.ones((5, 5))
order = 24
img_maxes = scipy.signal.order_filter(harris_response, domain, order)
out = np.zeros_like(harris_response)
mask = (harris_response == img_maxes)
out[mask] = harris_response[mask]

Ht = np.clip(out, 10000, np.inf)

handle = plt.imshow((Ht > 100000).astype('int'), interpolation='none')
#plt.show()

newThresh = 0.9*np.max(out)
mask2 = out > newThresh
newY, newX = np.nonzero(mask2)
score = out[mask2]

ind = np.flip(np.argsort(score, axis=0), axis=0)


print(ind)


tracking_points = np.column_stack((newX, newY))

print(score)

#I = lab1.load_lab_image('cornertest.png')
I_cross = I.copy()

for j, index in enumerate(ind):
    tx = tracking_points[index][1]
    ty = tracking_points[index][0]

    if tx < 10 or ty < 10:
        continue

    print(tx)
    print(ty)

    I_cross[tx-3:tx+3, ty-3:ty+3] = 255.0

    if j > 4:
        break

plt.imshow(I_cross, cmap="gray")
plt.show()

#max_img = ndimage.filters.maximum_filter(harris_thresholded, size=11)
#[row, col] = np.nonzero(harris_thresholded == max_img)

#plt.figure(1)
#plt.imshow(harris_thresholded, cmap="gray")
#plt.figure(2)
#plt.imshow(harris_thresholded, cmap="gray")
plt.show()

#plt.figure(1)
#plt.imshow(T_field[:, :, 0, 0], cmap="gray")
#plt.figure(2)
#plt.imshow(T_field[:, :, 1, 1], cmap="gray")
#plt.show()
