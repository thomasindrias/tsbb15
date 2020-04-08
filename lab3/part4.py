import lab3
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# 1. Compute mean for whole RIGHT image
# 2. Set uninteresting points to inf
# 3. For each interesting point, take that value minus one of LEFT image interesting point
# 4. Use joint min, return index

def find_interest_points(im):
    points = lab3.harris(im, 5, 5)
    points = lab3.non_max_suppression(points, 5)

    return points

def calculate_corr(ip_ind1, ip_ind2, rois1, rois2):
    (ip_ind1_rows, ip_ind1_cols) = ip_ind1
    (ip_ind2_rows, ip_ind2_cols) = ip_ind2
    len1 = len(ip_ind1_rows)
    len2 = len(ip_ind2_rows)
    match_matrix = np.inf * np.ones((len1, len2))

    corr1 = []
    corr2 = []
    
    for row in range(len1):
        rois1_row = rois1[row]
        for col in range(len2):
            rois2_col = rois2[col]
            error = np.sum(np.power(rois1_row-rois2_col, 2))
            
            match_matrix[row, col] = error

    (vals, ri, ci) = lab3.joint_min(match_matrix)

    for i, val in enumerate(vals):
        if val < 3000:
            a = ri[i]
            b = ci[i]
            corr1.append([ip_ind1_cols[a], ip_ind1_rows[a]])
            corr2.append([ip_ind2_cols[b], ip_ind2_rows[b]])

    return (corr1, corr2)

def get_corr(im1, im2):
    ip1 = find_interest_points(im1)
    ip2 = find_interest_points(im2)

    ip_ind1 = np.where(ip1 > 0.005)
    (rows1, cols1) = ip_ind1
    ip_ind2 = np.where(ip2 > 0.005)
    (rows2, cols2) = ip_ind2

    rois1 = lab3.cut_out_rois(im1, cols1, rows1, 7)
    rois2 = lab3.cut_out_rois(im2, cols2, rows2, 7)

    (corr1, corr2) = calculate_corr(ip_ind1, ip_ind2, rois1, rois2)

    corr1 = np.asarray(corr1).T
    corr2 = np.asarray(corr2).T

    #lab3.show_corresp(im1, im2, corr1, corr2)
    #plt.show()
    
    return(corr1, corr2)