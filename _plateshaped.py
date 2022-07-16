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
__all__ = ["platef_settings", "booth", "matya", "mccorm", "powersum", "zakharov"]

def platef_settings(dims=100):
    settings = {
        "booth": {"dims": 2, "bounds": [(-10, 10), (-10, 10)], "fmin": 0, "x*": [[1, 3]]},  # only 2 dims
        "matya": {"dims": 2, "bounds": [(-10, 10), (-10, 10)], "fmin": 0, "x*": [[0, 0]]},  # only 2 dims
        "mccorm": {"dims": 2, "bounds": [(-1.5, 4), (-3, 4)], "fmin": -1.9133, "x*": [[-0.54719, -1.54719]]},  # only 2 dims
        "powersum": {"dims": 4, "bounds": [(0, 4)] * 4, "fmin": None, "x*": []},
        "zakharov": {"dims": dims, "bounds": [(-5, 10)] * dims, "fmin": 0, "x*": [[0] * dims]},  # only 2 dims
    }
    return settings


def booth(X):
    """
    BOOTH FUNCTION
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/booth.html
    
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

        term1 = (x1 + 2 * x2 - 7) ** 2
        term2 = (2 * x1 + x2 - 5) ** 2

        y[i] = term1 + term2
    return y


def matya(X):
    """
    MATYAS FUNCTION
    
    The Matyas function has no local minima except the global one.
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/matya.html
    
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

        term1 = 0.26 * (x1 ** 2 + x2 ** 2)
        term2 = -0.48 * x1 * x2

        y[i] = term1 + term2
    return y


def mccorm(X):
    """
    MCCORMICK FUNCTION

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/mccorm.html
    
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

        term1 = np.sin(x1 + x2)
        term2 = (x1 - x2) ** 2
        term3 = -1.5 * x1
        term4 = 2.5 * x2

        y[i] = term1 + term2 + term3 + term4 + 1
    return y


def powersum(X, b=np.array([8, 18, 44, 114])):
    """
    POWER SUM FUNCTION

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/powersum.html
    
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

        outer = 0
        for ii in range(1, d+1):
            inner = 0
            for jj in range(0, d):
                xj = xx[jj]
                inner = inner + xj**ii
        outer = outer + (inner - b[ii-1])**2
        y[i] = outer
    return y

def zakharov(X):
    """
    ZAKHAROV FUNCTION

    The Zakharov function has no local minima except the global one.

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/zakharov.html

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

        sum1 = 0
        sum2 = 0

        for ii in range(1, d + 1):
            xi = xx[ii - 1]
            sum1 = sum1 + xi ** 2
            sum2 = sum2 + 0.5 * ii * xi

        y[i] = sum1 + sum2 ** 2 + sum2 ** 4
    return y
