import lab4
from estimate_T import estimate_T
from estimate_D import estimate_D
from get_HL import get_HL
from regularized_values import regularized_values
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.io as sio
import numpy as np

plt.set_cmap("gray")

# Algorithm Parameters
def diffusion():
    s = 0.05
    m = 0.0001
    steps = 100

    ksize = 1
    sigma = 0.5
    gradksize = 1
    gradsigma = 0.5

    L = lab4.get_cameraman() / 255.0

    L_init = np.copy(L)

    ax1 = plt.subplot(111)
    im1 = ax1.imshow(L)

    def update(i):
        nonlocal L

        T = estimate_T(L, gradksize, gradsigma, ksize, sigma)

        D = estimate_D(T, m)

        HL = get_HL(L)

        L = L + 0.5 * s + np.trace(D * HL, axis1=2, axis2=3)
        im1.set_data(L)

    ani = FuncAnimation(
        plt.gcf(), update, frames=range(steps), interval=5, repeat=False
    )
    plt.show()


diffusion()
