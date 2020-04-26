from regularized_values import regularized_values
from estimate_T import estimate_T
from get_HL import get_HL
import lab4
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt

plt.set_cmap("gray")


def inpainting():
    ksize = 4
    sigma = 1
    gradksize = 4
    gradsigma = 1

    alpha = 0.5
    lambda_ = 0.15

    im = lab4.get_cameraman()
    mask = (np.random.rand(im.shape[0], im.shape[1]) > 0.1) * 1.0
    g = im * mask
    u = g

    ax1 = plt.subplot(111)
    im1 = ax1.imshow(u)

    def init_func():
        pass

    def update(i):
        nonlocal u

        print(i)

        T = estimate_T(u, gradksize, gradsigma, ksize, sigma)
        H = get_HL(u)

        uxx = H[:, :, 0, 0]
        uxy = H[:, :, 1, 0]
        uyy = H[:, :, 1, 1]

        ux = T[:, :, 0, 0]
        uy = T[:, :, 1, 1]
        uxuy = T[:, :, 1, 0]

        HL = get_HL(u)

        diff_mask = mask*(u-g)

        div_u = (uxx * uy - 2 * uxy * uxuy + uyy * ux) / \
            np.power(np.sqrt(ux + uy), 3)

        u = u - alpha * (diff_mask - lambda_ * div_u)

        im1.set_data(u)

    ani = FuncAnimation(
        plt.gcf(),
        update,
        frames=range(2000),
        repeat=False,
        init_func=init_func,
    )
    plt.show()

inpainting()
