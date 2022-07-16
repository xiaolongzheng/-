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

__all__ = ["steepf_settings", "beale", "branin", "colville", "forretal08", "goldpr", "hart3",
           "hart4", "hart6", "permdb", "powell", "shekel", "stybtang"]

def steepf_settings(dims=100):
    settings = {
        "beale": {"dims": 2, "bounds": [(-4.5, 4.5), (-4.5, 4.5)], "fmin": 0, "x*": [[3, 0.5]]},  # only 2 dims
        "branin": {"dims": 2, "bounds": [(-5, 10), (0, 15)], "fmin": 0.397887,
                   "x*": [[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]]},
        "colville": {"dims": 4, "bounds": [(-10, 10)] * 4, "fmin": 0, "x*": [[1] * 4]},
        "forretal08": {"dims": 1, "bounds": [(0, 1)], "fmin": None, "x*": []},
        "goldpr": {"dims": 2, "bounds": [(-2, 2)] * 2, "fmin": 3, "x*": [[0, -1]]},
        "hart3": {"dims": 3, "bounds": [(0, 1)] * 3, "fmin": -3.86278, "x*": [[0.114614, 0.555649, 0.852547]]},
        "hart4": {"dims": 4, "bounds": [(0, 1)] * 4, "fmin": None, "x*": [None]},
        "hart6": {"dims": 6, "bounds": [(0, 1)] * 6, "fmin": -3.32237,
                  "x*": [[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]},
        "permdb": {"dims": dims, "bounds": [(-dims, dims)] * dims, "fmin": 0, "x*": [list(range(1, dims + 1))]},
        "powell": {"dims": dims, "bounds": [(-4, 5)] * dims, "fmin": 0, "x*": [[0] * dims]},
        "shekel": {"dims": 4, "bounds": [(0, 10)] * 4, "fmin": -10.5364, "x*": [[4] * 4]},
        "stybtang": {"dims": dims, "bounds": [(-5, 5)] * dims, "fmin": -39.16599 * dims, "x*": [[-2.903534] * dims]},
    }
    return settings


def beale(X):
    """
    BEALE FUNCTION
    
    The Beale function is multimodal, with sharp peaks at the corners of the input domain.
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/beale.html
    
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

        term1 = (1.5 - x1 + x1 * x2) ** 2
        term2 = (2.25 - x1 + x1 * x2 ** 2) ** 2
        term3 = (2.625 - x1 + x1 * x2 ** 3) ** 2

        y[i] = term1 + term2 + term3
    return y


def branin(X, a=1, b=5.1 / (4 * np.pi ** 2), c=5 / np.pi, r=6, s=10, t=1 / (8 * np.pi)):
    """
    BRANIN FUNCTION
    
    The Branin, or Branin-Hoo, function has three global minima.
    The recommended values of a, b, c, r, s and t are:
    a = 1, b = 5.1 ⁄ (4π2), c = 5 ⁄ π, r = 6, s = 10 and t = 1 ⁄ (8π).
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/branin.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:
    a = constant (optional), with default value 1
    b = constant (optional), with default value 5.1/(4*pi^2)
    c = constant (optional), with default value 5/pi
    r = constant (optional), with default value 6
    s = constant (optional), with default value 10
    t = constant (optional), with default value 1/(8*pi)

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]
        x1 = xx[0]
        x2 = xx[1]

        term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
        term2 = s * (1 - t) * np.cos(x1)

        y[i] = term1 + term2 + s
    return y


def colville(X):
    """
    COLVILLE FUNCTION

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/colville.html
    
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
        x3 = xx[2]
        x4 = xx[3]

        term1 = 100 * (x1 ** 2 - x2) ** 2
        term2 = (x1 - 1) ** 2
        term3 = (x3 - 1) ** 2
        term4 = 90 * (x3 ** 2 - x4) ** 2
        term5 = 10.1 * ((x2 - 1) ** 2 + (x4 - 1) ** 2)
        term6 = 19.8 * (x2 - 1) * (x4 - 1)

        y[i] = term1 + term2 + term3 + term4 + term5 + term6
    return y


def forretal08(X):
    """
    FORRESTER ET AL. (2008) FUNCTION
    
    This function is a simple one-dimensional test function.
    It is multimodal, with one global minimum,
    one local minimum and a zero-gradient inflection point.
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/forretal08.html
    
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
        fact1 = (6 * xx - 2) ** 2
        fact2 = np.sin(12 * xx - 4)

        y[i] = fact1 * fact2
    return y


def goldpr(X):
    """
    GOLDSTEIN-PRICE FUNCTION
    
    The Goldstein-Price function has several local minima.
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/goldpr.html
    
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

        fact1a = (x1 + x2 + 1) ** 2
        fact1b = 19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2
        fact1 = 1 + fact1a * fact1b

        fact2a = (2 * x1 - 3 * x2) ** 2
        fact2b = 18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2
        fact2 = 30 + fact2a * fact2b

        y[i] = fact1 * fact2
    return y


def hart3(X):
    """
    HARTMANN 3-DIMENSIONAL FUNCTION

    The 3-dimensional Hartmann function has 4 local minima.

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/hart3.html

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    alpha = np.array([[1.0], [1.2], [3.0], [3.2]])
    A = np.array([[3.0, 10, 30],
         [0.1, 10, 35],
         [3.0, 10, 30],
         [0.1, 10, 35]])
    P = 10 ** (-4) * np.array([[3689, 1170, 2673],
                      [4699, 4387, 7470],
                      [1091, 8732, 5547],
                      [381, 5743, 8828]])

    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]

        outer = 0
        for ii in range(0, 4):
            inner = 0
            for jj in range(0, 3):
                xj = xx[jj]
                Aij = A[ii, jj]
                Pij = P[ii, jj]
                inner = inner + Aij * (xj - Pij) ** 2

            new = alpha[ii] * np.exp(-inner)
            outer = outer + new
        y[i] = -outer
    return y


def hart4(X):
    """
    HARTMANN 4-DIMENSIONAL FUNCTION
    
    The 4-dimensional Hartmann function is multimodal.
    It is given here in the form of Picheny et al. (2012),
    having a mean of zero and a variance of one.
    The authors also add a small Gaussian error term to the output.
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/hart4.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2]).T
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 10 ** (-4) * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                               [2329, 4135, 8307, 3736, 1004, 9991],
                               [2348, 1451, 3522, 2883, 3047, 6650],
                               [4047, 8828, 8732, 5743, 1091, 381]])

    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]

        outer = 0
        for ii in range(0, 4):
            inner = 0
            for jj in range(0, 4):
                xj = xx[jj]
                Aij = A[ii, jj]
                Pij = P[ii, jj]
                inner = inner + Aij * (xj - Pij) ** 2
            new = alpha[ii] * np.exp(-inner)
            outer = outer + new
        y[i] = (1.1 - outer) / 0.839
    return y


def hart6(X):
    """
    HARTMANN 6-DIMENSIONAL FUNCTION

    The 6-dimensional Hartmann function has 6 local minima.

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/hart6.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    y = np.ones((X.shape[0],)) * np.inf
    alpha = np.array([1.0, 1.2, 3.0, 3.2]).T
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 10 ** (-4) * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                               [2329, 4135, 8307, 3736, 1004, 9991],
                               [2348, 1451, 3522, 2883, 3047, 6650],
                               [4047, 8828, 8732, 5743, 1091, 381]])
    for i in prange(X.shape[0]):
        xx = X[i, :]

        outer = 0
        for ii in range(0, 4):
            inner = 0
            for jj in range(0, 6):
                xj = xx[jj]
                Aij = A[ii, jj]
                Pij = P[ii, jj]
                inner = inner + Aij * (xj - Pij) ** 2
            new = alpha[ii] * np.exp(-inner)
            outer = outer + new
        y[i] = -(2.58 + outer) / 1.94
    return y


def permdb(X, b=0.5):
    """
    PERM FUNCTION d, beta

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/permdb.html
    
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
                inner = inner + (jj ** ii + b) * ((xj / jj) ** ii - 1)
            outer = outer + inner ** 2
        y[i] = outer
    return y


def powell(X):
    """
    POWELL FUNCTION

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/powell.html
    
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
        for ii in range(1, int(np.floor(d / 4))):
            term1 = (xx[4 * ii - 3 - 1] + 10 * xx[4 * ii - 2 - 1]) ** 2
            term2 = 5 * (xx[4 * ii - 1 - 1] - xx[4 * ii - 1]) ** 2
            term3 = (xx[4 * ii - 2 - 1] - 2 * xx[4 * ii - 1 - 1]) ** 4
            term4 = 10 * (xx[4 * ii - 3 - 1] - xx[4 * ii - 1]) ** 4
            sum = sum + term1 + term2 + term3 + term4
        y[i] = sum
    return y


def shekel(X, m=10):
    """
    SHEKEL FUNCTION
    
    The Shekel function has m local minima.
    Above are the recommended values of m, the β-vector and the C-matrix;
    β is an m-dimensional vector, and C is a 4-by-m-dimensional matrix.
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/shekel.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    y = np.ones((X.shape[0],)) * np.inf

    b = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5]).T
    C = np.array([[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                  [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                  [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                  [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]])
    for i in prange(X.shape[0]):
        xx = X[i, :]

        outer = 0
        for ii in range(0, m):
            bi = b[ii]
            inner = 0
            for jj in range(0, 4):
                xj = xx[jj]
                Cji = C[jj, ii]
                inner = inner + (xj - Cji) ** 2
            outer = outer + 1 / (inner + bi)
        y[i] = -outer
    return y


def stybtang(X):
    """
    STYBLINSKI-TANG FUNCTION
    
    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/stybtang.html
    
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
            new = xi ** 4 - 16 * xi ** 2 + 5 * xi
            sum = sum + new
        y[i] = sum / 2
    return y
