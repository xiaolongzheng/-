import numpy as np
from numba import njit, prange

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
__all__ = ["bowlf_settings", "boha1", "boha2", "boha3", "perm0db", "rothyp", "sumpow", "spheref", "sumsqu", "trid"]

def bowlf_settings(dims=100):
    settings = {
        "boha1": {"dims": 2, "bounds": [(-100, 100), (-100, 100)], "fmin": 0, "x*": [[0, 0]]},  # only 2 dims
        "boha2": {"dims": 2, "bounds": [(-100, 100), (-100, 100)], "fmin": 0, "x*": [[0, 0]]},  # only 2 dims
        "boha3": {"dims": 2, "bounds": [(-100, 100), (-100, 100)], "fmin": 0, "x*": [[0, 0]]},  # only 2 dims
        "perm0db": {"dims": dims, "bounds": [(-dims, dims)] * dims, "fmin": 0, "x*": [[1 / x for x in range(1, dims + 1)]]},
        "rothyp": {"dims": dims, "bounds": [(-65.536, 65.536)] * dims, "fmin": 0, "x*": [[0] * dims]},
        "spheref": {"dims": dims, "bounds": [(-5.12, 5.12)] * dims, "fmin": 0, "x*": [[0] * dims]},
        # "spherefmod": {"dims": 6, "bounds": [(0, 1)] * 6, "fmin": 0, "x*": [[0] * 6]},
        "sumpow": {"dims": dims, "bounds": [(-1, 1)] * dims, "fmin": 0, "x*": [[0] * dims]},
        # "sumsqu": {"dims": dims, "bounds": [(-5.12, 5.12)] * dims, "fmin": 0, "x*": [[0] * dims]},
        "sumsqu": {"dims": dims, "bounds": [(-10, 10)] * dims, "fmin": 0, "x*": [[0] * dims]},
        "trid": {"dims": dims, "bounds": [(-dims ** 2, dims ** 2)] * dims, "fmin": -dims * (dims + 4) * (dims - 1) / 6,
                 "x*": [[x * (dims + 1 - x) for x in range(1, dims + 1)]]},

    }
    return settings


def boha1(X):
    """
    BOHACHEVSKY FUNCTION 1
    
    The Bohachevsky functions all have the same similar bowl shape.
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/boha.html
    
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

        term1 = x1 ** 2
        term2 = 2 * x2 ** 2
        term3 = -0.3 * np.cos(3 * np.pi * x1)
        term4 = -0.4 * np.cos(4 * np.pi * x2)

        y[i] = term1 + term2 + term3 + term4 + 0.7
    return y


def boha2(X):
    """
    BOHACHEVSKY FUNCTION 2
    
    The Bohachevsky functions all have the same similar bowl shape.
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/
    
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

        term1 = x1 ** 2
        term2 = 2 * x2 ** 2
        term3 = -0.3 * np.cos(3 * np.pi * x1) * np.cos(4 * np.pi * x2)

        y[i] = term1 + term2 + term3 + 0.3
    return y


def boha3(X):
    """
    BOHACHEVSKY FUNCTION 3
    
    The Bohachevsky functions all have the same similar bowl shape.
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/Code/boha3m.html
    
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

        term1 = x1 ** 2
        term2 = 2 * x2 ** 2
        term3 = -0.3 * np.cos(3 * np.pi * x1 + 4 * np.pi * x2)

        y[i] = term1 + term2 + term3 + 0.3
    return y


def perm0db(X, b=10):
    """
    PERM FUNCTION 0, d, beta
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/perm0db.html
    
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
        for ii in range(1, d + 1):
            inner = 0
            for jj in range(1, d + 1):
                xj = xx[jj - 1]
                inner = inner + (jj + b) * (xj ** ii - (1 / jj) ** ii)
            outer = outer + inner ** 2
        y[i] = outer
    return y


def rothyp(X):
    """
    ROTATED HYPER-ELLIPSOID FUNCTION
    
    The Rotated Hyper-Ellipsoid function is continuous, convex and unimodal.
    It is an extension of the Axis Parallel Hyper-Ellipsoid function,
    also referred to as the Sum Squares function.
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/rothyp.html
    
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
        for ii in range(1, d + 1):
            inner = 0
            for jj in range(1, ii + 1):
                xj = xx[jj - 1]
                inner = inner + xj ** 2
            outer = outer + inner
        y[i] = outer
    return y

@njit(parallel=True)
def spheref0(X):
    """
    SPHERE FUNCTION
    
    The Sphere function has dims local minima except for the global one.
    It is continuous, convex and unimodal.
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/spheref.html
    
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
        for ii in range(0, d):
            xi = xx[ii]
            sum = sum + xi ** 2
        y[i] = sum
    return y

@njit(parallel=True)
def spheref(X):
    """
    SPHERE FUNCTION

    The Sphere function has dims local minima except for the global one.
    It is continuous, convex and unimodal.

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/spheref.html

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
    for i in range(X.shape[0]):
        xx = X[i, :]

        sum = np.zeros((d,))
        for ii in prange(0, d):
            xi = xx[ii]
            sum[ii] = xi ** 2
        y[i] = np.sum(sum)
    return y


def spherefmod(X):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    SPHERE FUNCTION, MODIFIED
    
    The Sphere function has dims local minima except for the global one.
    It is continuous, convex and unimodal.
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/spheref.html
    
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

        sum = 0
        for ii in range(1, 6 + 1):
            xi = xx[ii - 1]
            sum = sum + (xi ** 2) * (2 ** ii)

        y[i] = (sum - 1745) / 899
    return y


def sumpow(X):
    """
    SUM OF DIFFERENT POWERS FUNCTION
    
    The Sum of Different Powers function is unimodal.
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/sumpow.html
    
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
            new = (np.abs(xi)) ** (ii + 1)
            sum = sum + new
        y[i] = sum
    return y


def sumsqu(X):
    """
    SUM SQUARES FUNCTION
    
    The Sum Squares function, also referred to as the Axis Parallel Hyper-Ellipsoid function,
    has no local minimum except the global one.
    It is continuous, convex and unimodal.
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/sumsqu.html
    
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
            sum = sum + ii * xi ** 2
        y[i] = sum
    return y


def trid(X):
    """
    TRID FUNCTION
    
    The Trid function has no local minimum except the global one.
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/trid.html
    
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

        sum1 = (xx[0] - 1) ** 2
        sum2 = 0
        for ii in range(1, d):
            xi = xx[ii]
            xold = xx[ii - 1]
            sum1 = sum1 + (xi - 1) ** 2
            sum2 = sum2 + xi * xold
        y[i] = sum1 - sum2
    return y
