

# 2
# Use Harris detector to find points.
# There are 8 unknown variables in transformation matrix so we need n >= 8.

# 3
# We must be able to find corresponding points between the two images. 

# 4
# Use Optic flow to find correspondences.


import lab3
from part4 import get_corr
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import random

plt.set_cmap("gray")

[im1, im2] = lab3.load_stereo_pair()

(corr1, corr2) = get_corr(im1, im2)

best_inlier_count = 0
best_F = None
best_std = np.inf

for i in range(60):
    random_index = random.sample(range(len(corr1[0])), 8)

    corr1_rand = corr1[:, random_index]
    corr2_rand = corr2[:, random_index]

    F = lab3.fmatrix_stls(corr1_rand, corr2_rand)

    d = lab3.fmatrix_residuals(F, corr1_rand, corr2_rand)

    thresh = 1
    inlier_count = 0

    for j in range(8):
        norm = np.linalg.norm(d[:, j])
        if norm < thresh:
            inlier_count = inlier_count + 1
    
    std = np.std(np.linalg.norm(d, axis=0))
    
    if (inlier_count > best_inlier_count) or (inlier_count == best_inlier_count and std < best_std):
        best_std = std
        best_inlier_count = inlier_count
        best_F = F
        best_corr1 = corr1_rand
        best_corr2 = corr2_rand

inlier_ratio = best_inlier_count / 8

print("INLIER RATIO", inlier_ratio)

lab3.show_corresp(im1, im2, best_corr1, best_corr2)
plt.show()

plt.figure(1)
lab3.plot_eplines(F, corr2, im1.shape)

plt.figure(2)
lab3.plot_eplines(F.T, corr1, im2.shape)


plt.show()
