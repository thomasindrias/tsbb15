import lab4
from estimate_T import estimate_T
from estimate_D import estimate_D
from regularized_values import regularized_values
from matplotlib import pyplot as plt
import scipy.io as sio
import numpy as np

ksize = 15
sigma = 2
gradksize = 15
gradsigma = 2

I = lab4.get_cameraman()

T = estimate_T(I, gradksize, gradsigma, ksize, sigma)
sio.savemat('T.mat', {'data': T})

D = estimate_D(T, 0.1)
#plt.imshow(T[:, :, 0, 0], cmap="gray")
#plt.show()
