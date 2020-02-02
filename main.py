import regularized_values
import estimate_T
import estimate_e
import numpy as np
import scipy
from matplotlib import pyplot as plt

I = plt.imread('frame1.png')
J = plt.imread('frame2.png')

I = 0.299*I[:, :, 0] + 0.587*I[:, :, 1] + 0.114*I[:, :, 2]
J = 0.299*J[:, :, 0] + 0.587*J[:, :, 1] + 0.114*J[:, :, 2]

(Ig, Jg, Jgdx, Jgdy) = regularized_values.regularized_values(I, J, 17, 6.0)

# plt.figure(1)
#plt.imshow(Jgdx, cmap="gray")

T = estimate_T.estimate_T(Jgdx, Jgdy, 600, 600, [512, 512])
e = estimate_e.estimate_e(Ig, Jg, Jgdx, Jgdy, 600, 600, [512, 512])

# plt.figure(1)
#plt.imshow(e[:, :, 1], cmap="gray")
# plt.show()

# Find an appropiate value, d
d_tot = 0

plt.imshow(T[50:1000, 50:1000, 0, 0], cmap="gray")
plt.show()

# d = T^-1 * e
d = np.linalg.solve(T[100, 100], e[100, 100])
