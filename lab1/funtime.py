import numpy as np
from lk_tracker import lk_tracker
from matplotlib import pyplot as plt
import PIL.Image
from scipy import ndimage
from regularized_values import regularized_values
from harris import harris, orientation_tensor, non_max_suppression
import lab1
import scipy

def load_lab_image(filename):
    """Load a grayscale image by filename from the CVL image directory

    Example:
    >>> img = load_lab_image('cornertest.png')
    """

    return np.asarray(PIL.Image.open(filename).convert('L'))


#tracking_points = [[346, 428], [100, 200], [200, 200]]

I = load_lab_image('frame1.png')
I_cross = I.copy()

T_field = orientation_tensor(I, 11, 2.0, 11, 1.6)

harris_response = harris(T_field, 0.05)

#harris_thresholded = harris_response > 3000
#harris_thresholded = harris_thresholded * 1

#[ht1, ht2] = np.nonzero(harris_thresholded)

best_features = non_max_suppression(harris_response, (5, 5))

Ht = np.clip(best_features, 15000, np.inf)
handle = plt.imshow((Ht > 20000).astype('int'), interpolation='none')
plt.show()

newThresh = 0.8*np.max(best_features)
mask2 = best_features > newThresh
newX, newY = np.nonzero(mask2)
score = best_features[mask2]

ind = np.flip(np.argsort(score, axis=0), axis=0)

print("IND", ind)
print("SCORE", score)

tracking_points = np.column_stack(((newX[ind]), newY[ind]))
print(tracking_points)

new_tracking_points = []

for p in tracking_points:
    if p[0] < 10 or p[1] < 10:
        continue
    new_tracking_points.append(p)

new_tracking_points = np.asarray(new_tracking_points)
tracking_points = new_tracking_points[0:5]

print("NEW",new_tracking_points)

#print(ht1, ht2)

#max_img = ndimage.filters.maximum_filter(harris_thresholded, size=3)
#mask = (harris_thresholded == max_img)
#harris_thresholded *= mask
#plt.imshow(harris_thresholded, cmap="gray")
#plt.show()
#[ht1, ht2] = np.nonzero(max_img)
#[row, col] = np.nonzero(harris_thresholded == max_img)



#tracking_points = np.column_stack((ht1, ht2))
#print(tracking_points[1])
#tracking_points = tracking_points[((tracking_points.shape[0]//2)-3):((tracking_points.shape[0]//2)+3)]

#print(tracking_points)


#tracking_points = [[316, 494]]

for tracking_point in tracking_points:
    tx = tracking_point[0]
    ty = tracking_point[1]
    I_cross[tx-3:tx+3, ty-3:ty+3] = 255.0

output_imgs = []

img_count = 2

while img_count <= 2:
    J = load_lab_image('frame'+str(img_count)+'.png')

    (Ig, Jg, Jgdx, Jgdy) = regularized_values(I, J, 17, 3.0)

    d_tots = []
    for point in tracking_points:
        d_tots.append(lk_tracker(Ig, Jg, Jgdx, Jgdy, point))

    J_cross = J.copy()

    for i in range(len(d_tots)):
        txx = tracking_points[i][0] + int(np.round(d_tots[i][1]))
        tyy = tracking_points[i][1] + int(np.round(d_tots[i][0]))
        J_cross[txx-3:txx+3, tyy-3:tyy+3] = 255.0

    output_imgs.append(J_cross)

    img_count = img_count + 1


#plt.imshow(I_cross, cmap="gray")
#plt.show()
plt.imshow(output_imgs[0], cmap="gray")
plt.show()
plt.imshow(output_imgs[1], cmap="gray")
plt.show()
