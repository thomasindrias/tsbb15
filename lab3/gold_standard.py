import lab3
import matplotlib.pyplot as plt
from part4 import get_corr
import numpy as np
from scipy import signal
import random
from scipy.optimize import least_squares


def gold_standard(corr1, corr2, F=None):
    random_index = random.sample(range(len(corr1[1])), 8)

    corr1_rand = corr1[:, random_index]
    corr2_rand = corr2[:, random_index]

    if not isinstance(F, np.ndarray):
        F = lab3.fmatrix_stls(corr1_rand, corr2_rand)

    # P2 = [I|0]
    (C1, C2) = lab3.fmatrix_cameras(F)

    tri_points = []

    for i in range(corr1.shape[1]):
        p = lab3.triangulate_optimal(C1, C2, corr1[:, i], corr2[:, i])
        tri_points.append(p)

    tri_points = np.asarray(tri_points)

    # print(tri_points)

    param1 = np.hstack((C1.ravel(), tri_points.ravel()))

    #print(lab3.fmatrix_residuals_gs(param1, corr1_rand, corr2_rand))

    def cost(x):
        return lab3.fmatrix_residuals_gs(x, corr1, corr2)

    D2 = least_squares(cost, param1)

    C1_refined = np.reshape(D2.x[0:12], [3, 4])

    F_new = lab3.fmatrix_from_cameras(C1_refined, C2)

    # plt.figure(1)
    # plt.imshow(im1)
    #lab3.plot_eplines(F_new, corr2_rand, (im1.shape[1], im1.shape[0]))

    # plt.figure(2)
    # plt.imshow(im2)
    #lab3.plot_eplines(F_new.T, corr1_rand, (im2.shape[1], im2.shape[0]))

    # plt.show()

    return (corr1_rand, corr2_rand, F_new)
