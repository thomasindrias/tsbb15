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
    s = 0.2
    m = 0.3
    steps = 100

    ksize = 9
    sigma = 1.5
    gradksize = 9
    gradsigma = 1.5

    L = lab4.get_cameraman()
    gaussian = np.random.normal(0, 10, (L.shape[0], L.shape[1]))
    L = L + gaussian

    # L = lab4.make_circle(128, 128, 120) * 255
    # gaussian = np.random.normal(0, 10, (L.shape[0], L.shape[1]))
    # L = L + gaussian

    fig1 = plt.figure(1)
    fig1.suptitle("Original Image w/ noise")
    plt.imshow(L)

    fig2 = plt.figure(2)
    fig2.suptitle("Enhanced Image")

    L_init = np.copy(L)

    ax1 = plt.subplot(111)
    im1 = ax1.imshow(L)

    def init_func():
        pass

    def update(i):
        nonlocal L

        T = estimate_T(L, gradksize, gradsigma, ksize, sigma)

        D = estimate_D(T, m)

        HL = get_HL(L)

        L = L + 0.5 * s * np.trace(D * HL, axis1=2, axis2=3)

        print("Steps:", i)

        im1.set_data(L)

    ani = FuncAnimation(
        plt.gcf(),
        update,
        frames=range(steps),
        interval=5,
        repeat=False,
        init_func=init_func,
    )
    plt.show()


diffusion()
