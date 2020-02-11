import numpy as np

def LK_errors(I, J, Jv):
    error1 = np.linalg.norm(I-J)
    error2 = np.linalg.norm(I-Jv)
    print("||J(x) - I(x)||", error1)
    print("||J(x + v) - I(x)||", error2)
    return (error1, error2)
