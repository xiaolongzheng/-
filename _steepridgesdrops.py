import numpy as np
from numba import prange

"""
Templates

    def constr_f(x):
       return np.array(x[0] + x[1])

    # not vectoriable
    def constr_f(X):
       y = np.ones((X.shape[0], )) * np.inf
       for i in prange(X.shape[0]):
          x = X[i, :]
          y[i] = x[0] + x[1]
       return y

    #  vectoriable
    def constr_f(X):
       return np.array(X[:, 0] + X[:, 1])
"""
__all__ = ["steepf_settings", "dejong5", "easom", "michal"]

def steepf_settings(dims=100):
    settings = {
        "dejong5": {"dims": 2, "bounds": [(-65.536, 65.536), (-65.536, 65.536)], "fmin": None, "x*": []},  # only 2 dims
        "easom": {"dims": 2, "bounds": [(-100, 100), (-100, 100)], "fmin": -1, "x*": [[np.pi, np.pi]]},
        "michal": {"dims": dims, "bounds": [(0, np.pi)] * dims, "fmin": None, "x*": []},  # various
    }
    return settings


def dejong5(X):
    """
    DE JONG FUNCTION N. 5
    
    The fifth function of De Jong is multimodal, with very sharp drops on a mainly flat surface.
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/dejong5.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]
        x1 = xx[0]
        x2 = xx[1]

        sum = 0
        A = np.zeros((2, 25))
        a = np.array([-32, -16, 0, 16, 32])
        A[0] = np.tile(a, (1, 5))
        ar = np.tile(a, (5, 1))
        ar = ar.reshape(1, ar.size, order="F")
        A[1] = ar

        for ii in range(1, 25 + 1):
            a1i = A[0, ii - 1]
            a2i = A[1, ii - 1]
            term1 = ii
            term2 = (x1 - a1i) ** 6
            term3 = (x2 - a2i) ** 6
            new = 1 / (term1 + term2 + term3)
            sum = sum + new

        y[i] = 1 / (0.002 + sum)
    return y


def easom(X):
    """
    EASOM FUNCTION
    
    The Easom function has several local minima.
    It is unimodal, and the global minimum has a small area relative to the search space.
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/easom.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]
        x1 = xx[0]
        x2 = xx[1]

        fact1 = -np.cos(x1) * np.cos(x2)
        fact2 = np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)

        y[i] = fact1 * fact2
    return y


def michal(X, m=10):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    MICHALEWICZ FUNCTION
    
    The Michalewicz function has d! local minima, and it is multimodal.
    The parameter m defines the steepness of they valleys and ridges;
    a larger m leads to a more difficult search.
    The recommended value of m is m = 10.
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/michal.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    d = X.shape[1]
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]

        sum = 0
        for ii in range(1, d + 1):
            xi = xx[ii - 1]
            new = np.sin(xi) * (np.sin(ii * xi ** 2 / np.pi)) ** (2 * m)
            sum = sum + new
        y[i] = -sum
    return y
