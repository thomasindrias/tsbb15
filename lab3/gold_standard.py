import lab3
import matplotlib.pyplot as plt
from part4 import get_corr
import numpy as np
from scipy import signal
import random

plt.set_cmap("gray")

[im1, im2] = lab3.load_stereo_pair()

(corr1, corr2) = get_corr(im1, im2)

random_index = random.sample(range(len(corr1[1])), 8)

corr1_rand = corr1[:, random_index]
corr2_rand = corr2[:, random_index]

F = lab3.fmatrix_stls(corr1_rand, corr2_rand)

# P2 = [I|0]
(C1, C2) = lab3.fmatrix_cameras(F)

tri_points = []

for i in range(corr1_rand.shape[1]):
    p = lab3.triangulate_optimal(C1, C2, corr1_rand[:, i], corr2_rand[: ,i])
    tri_points.append([p])

tri_points = np.asarray(tri_points)

print(tri_points)

#TODO 
# step (iii) least square thing