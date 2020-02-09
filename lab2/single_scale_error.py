import numpy as np
from matplotlib import pyplot as plt


def single_scale_error(I, J):
    error = np.linalg.norm(I-J)

    return(error)
