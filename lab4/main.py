import lab4
from add_noise import add_noise
from estimate_T import estimate_T
from estimate_D import estimate_D
from get_HL import get_HL
from add_noise import add_noise
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.io as sio
import numpy as np

plt.set_cmap("gray")

# Algorithm Parameters


def diffusion():
    s = 0.4
    m = 0.3
    steps = 500

    ksize = 9
    sigma = 1.5
    gradksize = 9
    gradsigma = 1.5

    errors = []

    L = lab4.get_cameraman()
    L_noise = add_noise(L, 10)
    #L_noise = add_noise(L, 8)
    #L_noise = add_noise(L, 12)

    # ----------------- Multiplicative noise -------------------
    #L_log = np.log(L)
    #L_noise = add_noise(L_log, 40)
    #L_noise = np.exp(L_noise)

    # L = lab4.make_circle(128, 128, 120) * 255
    # gaussian = np.random.normal(0, 10, (L.shape[0], L.shape[1]))
    # L = L + gaussian

    fig1 = plt.figure(1)
    fig1.suptitle("Original Image w/ noise")
    plt.imshow(L)

    fig2 = plt.figure(2)
    fig2.suptitle("Enhanced Image")

    L_init = np.copy(L_noise)

    ax1 = plt.subplot(111)
    im1 = ax1.imshow(L)

    def init_func():
        pass

    def update(i):
        nonlocal L_noise

        T = estimate_T(L_noise, gradksize, gradsigma, ksize, sigma)

        D = estimate_D(T, m)

        HL = get_HL(L_noise)

        L_noise = L_noise + 0.5 * s * np.trace(D * HL, axis1=2, axis2=3)

        errors.append(np.sum(np.abs(L_noise - L)))
        # print(np.sum(np.abs(L_noise - L)))

        print("Steps:", i)

        im1.set_data(L_noise)

    ani = FuncAnimation(
        plt.gcf(),
        update,
        frames=range(steps),
        repeat=False,
        init_func=init_func,
    )
    plt.show()

    plt.plot(errors)
    plt.show()


diffusion()
