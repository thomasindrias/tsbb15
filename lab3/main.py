import lab3
from part4 import get_corr
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import random
from ransac import ransac
from gold_standard import gold_standard
import time

plt.set_cmap("gray")

[im1, im2] = lab3.load_stereo_pair()

(corr1, corr2) = get_corr(im1, im2)

print("Total points: ", corr1.shape[1])

# RANSAC
start = time.time()

(best_corr1, best_corr2, best_F) = ransac(corr1, corr2, 2000)

end = time.time()
ransac_time = end-start

lab3.show_corresp(im1, im2, corr1, corr2)
plt.show()

lab3.show_corresp(im1, im2, best_corr1, best_corr2)
plt.show()

plt.figure(1)
plt.imshow(im1)
lab3.plot_eplines(best_F, best_corr2, (im1.shape[1], im1.shape[0]))

plt.figure(2)
plt.imshow(im2)
lab3.plot_eplines(best_F.T, best_corr1, (im2.shape[1], im2.shape[0]))

plt.show()

# GOLD STANDARD
start = time.time()

(gs_corr1, gs_corr2, gs_F) = gold_standard(best_corr1, best_corr2)

end = time.time()
gs_time = end-start

plt.figure(1)
plt.imshow(im1)
lab3.plot_eplines(gs_F, gs_corr2, (im1.shape[1], im1.shape[0]))

plt.figure(2)
plt.imshow(im2)
lab3.plot_eplines(gs_F.T, gs_corr1, (im2.shape[1], im2.shape[0]))

plt.show()

# Verification
print("--------------- VERIFICATION -----------------")
ransac_d = lab3.fmatrix_residuals(best_F, corr1, corr2)
thresh = 1
inlier_count = 0

for j in range(ransac_d.shape[1]):
    norm = np.linalg.norm(ransac_d[:, j])
    if norm < thresh:
        inlier_count = inlier_count + 1

print("RANSAC INLIERS: ", inlier_count)
print("RANSAC TIME: ", ransac_time, "s")
print("RANSAC MEAN RESIDUAL:", np.abs(np.mean(ransac_d)))

print("")

gs_d = lab3.fmatrix_residuals(gs_F, corr1, corr2)
thresh = 1
inlier_count = 0

for j in range(gs_d.shape[1]):
    norm = np.linalg.norm(gs_d[:, j])
    if norm < thresh:
        inlier_count = inlier_count + 1

print("GOLD STANDARD INLIERS:", inlier_count)
print("GOLD STANDARD TIME:", gs_time, "s")
print("GOLD STANDARD MEAN RESIDUAL:", np.abs(np.mean(gs_d)))
