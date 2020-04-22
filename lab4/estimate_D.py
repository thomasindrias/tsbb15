import scipy.linalg as la
import numpy as np


def estimate_D(T, m):
    Tr = T[:, :, 0, 0] + T[:, :, 1, 1]
    Det = (T[:, :, 1, 1] * T[:, :, 0, 0]) - (T[:, :, 1, 0] * T[:, :, 0, 1])

    lambda_1 = Tr / 2 + np.sqrt(np.power(Tr, 2) / 4 - Det)
    lambda_2 = Tr / 2 - np.sqrt(np.power(Tr, 2) / 4 - Det)

    e1 = np.array([lambda_1 - T[:, :, 1, 1], T[:, :, 1, 0]])[:, None, :, :]
    e1 = e1 / np.linalg.norm(e1, axis=0)
    
    e2 = np.array([lambda_2 - T[:, :, 1, 1], T[:, :, 1, 0]])[:, None, :, :]
    e2 = e2 / np.linalg.norm(e2, axis=0)

    alpha_1 = np.exp(-lambda_1 / m)
    alpha_2 = np.exp(-lambda_2 / m)

    D = alpha_1 * e1 * np.transpose(e1, (1, 0, 2, 3)) + alpha_2 * e2 * np.transpose(
        e2, (1, 0, 2, 3)
    )

    return np.transpose(D, (2, 3, 0, 1))
