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

__all__ = ["manyf_settings", "ackley", "bukin6", "crossit", "drop", "egg", "grlee12","griewank", "holder", "langer",
           "levy", "levy13", "rastr", "schaffer2", "schaffer4", "schwef", "shubert"]

def manyf_settings(dims=100):
    settings = {
        "ackley": {"dims": dims, "bounds": [(-32.768, 32.768)] * dims, "fmin": 0, "x*": [[0] * dims]},
        "bukin6": {"dims": 2, "bounds": [(-15, -5), (-3, 3)], "fmin": 0, "x*": [[-10, 1]]},  # only 2 dims
        "crossit": {"dims": 2, "bounds": [(-10, 10), (-10, 10)], "fmin": -2.06261,
                    "x*": [[1.3491, -1.3491], [1.3491, 1.3491], [-1.3491, 1.3491], [-1.3491, -1.3491]]},
        "drop": {"dims": 2, "bounds": [(-5.12, 5.12), (-5.12, 5.12)], "fmin": -1, "x*": [[0, 0]]},
        "egg": {"dims": 2, "bounds": [(-512, 512), (-512, 512)], "fmin": -959.6407, "x*": [[512, 404.2319]]},
        "grlee12": {"dims": 1, "bounds": [(0.5, 2.5)], "fmin": None, "x*": []},
        "griewank": {"dims": dims, "bounds": [(-600, 600)] * dims, "fmin": 0, "x*": [[0] * dims]},
        "holder": {"dims": 2, "bounds": [(-10, 10), (-10, 10)], "fmin": -19.2085,
                   "x*": [[8.05502, 9.66459], [8.05502, -9.66459], [-8.05502, 9.66459], [-8.05502, -9.66459]]},
        "langer": {"dims": 2, "bounds": [(-10, 10), (-10, 10)], "fmin": None, "x*": []},
        "levy": {"dims": dims, "bounds": [(-10, 10)] * dims, "fmin": 0, "x*": [[1] * dims]},
        "levy13": {"dims": 2, "bounds": [(-10, 10), (-10, 10)], "fmin": 0, "x*": [[1, 1]]},
        "rastr": {"dims": dims, "bounds": [(-5.12, 5.12)] * dims, "fmin": 0, "x*": [[0] * dims]},
        "schaffer2": {"dims": 2, "bounds": [(-100, 100), (-100, 100)], "fmin": 0, "x*": [[0, 0]]},
        "schaffer4": {"dims": 2, "bounds": [(-100, 100), (-100, 100)], "fmin": None, "x*": []},
        "schwef": {"dims": dims, "bounds": [(-500, 500)] * dims, "fmin": 0, "x*": [[420.9687] * dims]},
        # "shubert": {"dims": 2, "bounds": [(-5.12, 5.12),(-5.12, 5.12)], "fmin": -186.7309, "x*": []},
        "shubert": {"dims": 2, "bounds": [(-10, 10), (-10, 10)], "fmin": -186.7309, "x*": []},
    }
    return settings


# @njit("float64[:](float64[:,:], float64, float64, float64)", parallel=True)
@njit(parallel=True)
def ackley(X, a=20.0, b=0.2, c=2 * np.pi):
    """
    ACKLEY FUNCTION

    The Ackley function is widely used for testing optimization algorithms.
    In its two-dimensional form, it is characterized by a nearly flat outer region, and a large hole at the centre.
    The function poses a risk for optimization algorithms, particularly hillclimbing algorithms,
    to be trapped in one of its many local minima.
    Recommended variable values are: a = 20, b = 0.2 and c = 2π.

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/ackley.html

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:
    a = constant (optional), with default value 20
    b = constant (optional), with default value 0.2
    c = constant (optional), with default value 2*pi

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    if X.ndim<2:
        Xt = np.expand_dims(X, 0)
    else:
        Xt = X
    X = Xt
    d = X.shape[1]
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]

        sum1 = 0
        sum2 = 0
        for ii in range(0, d):
            xi = xx[ii]
            sum1 = sum1 + xi ** 2
            sum2 = sum2 + np.cos(c * xi)

        term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)

        y[i] = term1 + term2 + a + np.exp(1)
    return y


# @njit("float64[:](float64[:,:])", parallel=True)
@njit(parallel=True)
def bukin6(X):
    """
    BUKIN FUNCTION N. 6

    The sixth Bukin function has many local minima, all of which lie in a ridge.

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/bukin6.html

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    if X.ndim<2:
        Xt = np.expand_dims(X, 0)
    else:
        Xt = X
    X = Xt
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]

        x1 = xx[0]
        x2 = xx[1]

        term1 = 100 * np.sqrt(np.abs(x2 - 0.01 * x1 ** 2))
        term2 = 0.01 * np.abs(x1 + 10)

        y[i] = term1 + term2
    return y


@njit(parallel=True)
def crossit(X):
    """
    CROSS-IN-TRAY FUNCTION

    The Cross-in-Tray function has multiple global minima.

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/crossit.html

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    if X.ndim<2:
        Xt = np.expand_dims(X, 0)
    else:
        Xt = X
    X = Xt
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]

        x1 = xx[0]
        x2 = xx[1]

        fact1 = np.sin(x1) * np.sin(x2)
        fact2 = np.exp(abs(100 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi))

        y[i] = -0.0001 * (np.abs(fact1 * fact2) + 1) ** 0.1
    return y


@njit(parallel=True)
def drop(X):
    """
    DROP-WAVE FUNCTION

    The Drop-Wave function is multimodal and highly complex.

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/drop.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    if X.ndim<2:
        Xt = np.expand_dims(X, 0)
    else:
        Xt = X
    X = Xt
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]

        x1 = xx[0]
        x2 = xx[1]

        frac1 = 1 + np.cos(12 * np.sqrt(x1 ** 2 + x2 ** 2))
        frac2 = 0.5 * (x1 ** 2 + x2 ** 2) + 2

        y[i] = -frac1 / frac2
    return y


@njit(parallel=True)
def egg(X):
    """
    EGGHOLDER FUNCTION

    The Eggholder function is a difficult function to optimize, because of the large number of local minima.

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/egg.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    if X.ndim<2:
        Xt = np.expand_dims(X, 0)
    else:
        Xt = X
    X = Xt
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        x = X[i, :]

        x1 = x[0]
        x2 = x[1]

        term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47)))
        term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

        y[i] = term1 + term2
    return y


# @njit(parallel=True)
def grlee12(X):
    """
    GRAMACY & LEE (2012) FUNCTION

    This is a simple one-dimensional test function.

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/grlee12.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    if X.ndim<2:
        Xt = np.expand_dims(X, 0)
    else:
        Xt = X
    X = Xt
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        x = X[i, :]

        term1 = np.sin(10 * np.pi * x) / (2 * x)
        term2 = (x - 1) ** 4

        y[i] = term1 + term2
    return y


@njit(parallel=True)
def griewank(X):
    """
    GRIEWANK FUNCTION

    The Griewank function has many widespread local minima, which are regularly distributed.

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/griewank.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    if X.ndim<2:
        Xt = np.expand_dims(X, 0)
    else:
        Xt = X
    X = Xt
    d = X.shape[1]
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]

        sum = 0
        prod = 1

        for ii in range(1, d + 1):
            xi = xx[ii - 1]
            sum = sum + (xi ** 2) / 4000
            prod = prod * np.cos(xi / np.sqrt(ii + 1))

        y[i] = sum - prod + 1
    return y


@njit(parallel=True)
def holder(X):
    """
    HOLDER TABLE FUNCTION

    The Holder Table function has many local minima, with four global minima.

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/holder.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    if X.ndim<2:
        Xt = np.expand_dims(X, 0)
    else:
        Xt = X
    X = Xt
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]

        x1 = xx[0]
        x2 = xx[1]

        fact1 = np.sin(x1) * np.cos(x2)
        fact2 = np.exp(np.abs(1 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi))

        y[i] = -np.abs(fact1 * fact2)
    return y


@njit(parallel=True)
def langer(X, m=5, c=np.array([1, 2, 5, 2, 3]), A=np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])):
    """
    LANGERMANN FUNCTION

    The Langermann function is multimodal, with many unevenly distributed local minima.

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/langer.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    if X.ndim<2:
        Xt = np.expand_dims(X, 0)
    else:
        Xt = X
    X = Xt
    d = X.shape[1]
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]

        outer = 0
        for ii in range(0, m):
            inner = 0
            for jj in range(0, d):
                xj = xx[jj]
                Aij = A[ii, jj]
                inner = inner + (xj - Aij) ** 2
            new = c[ii] * np.exp(-inner / np.pi) * np.cos(np.pi * inner)
            outer = outer + new
        y[i] = outer
    return y


@njit(parallel=True)
def levy0(X):
    """
    LEVY FUNCTION

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/Code/levym.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    if X.ndim<2:
        Xt = np.expand_dims(X, 0)
    else:
        Xt = X
    X = Xt
    d = X.shape[1]
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]

        w = np.ones_like(xx) * np.inf
        for ii in range(0, d):
            w[ii] = 1 + (xx[ii] - 1) / 4

        term1 = (np.sin(np.pi * w[0])) ** 2
        term3 = (w[-1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[-1])) ** 2)

        sum = 0
        for ii in range(0, d - 1):
            wi = w[ii]
            new = (wi - 1) ** 2 * (1 + 10 * (np.sin(np.pi * wi + 1)) ** 2)
            sum = sum + new
        y[i] = term1 + sum + term3
    return y


@njit(parallel=True)
def levy(X):
    """
    LEVY FUNCTION

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/Code/levym.html

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    if X.ndim < 2:
        Xt = np.expand_dims(X, 0)
    else:
        Xt = X
    X = Xt
    d = X.shape[1]
    y = np.ones((X.shape[0],)) * np.inf
    for i in range(X.shape[0]):
        xx = X[i, :]

        w = np.ones_like(xx) * np.inf
        for ii in prange(0, d):
            w[ii] = 1 + (xx[ii] - 1) / 4

        term1 = (np.sin(np.pi * w[0])) ** 2
        term3 = (w[-1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[-1])) ** 2)

        sum = np.zeros((d-1,))
        for ii in prange(0, d - 1):
            wi = w[ii]
            new = (wi - 1) ** 2 * (1 + 10 * (np.sin(np.pi * wi + 1)) ** 2)
            sum[ii] = + new
        y[i] = term1 + np.sum(sum) + term3
    return y

@njit(parallel=True)
def levy13(X):
    """
    LEVY FUNCTION N. 13

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/Code/levy13m.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    if X.ndim<2:
        Xt = np.expand_dims(X, 0)
    else:
        Xt = X
    X = Xt
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]

        x1 = xx[0]
        x2 = xx[1]

        term1 = (np.sin(3 * np.pi * x1)) ** 2
        term2 = (x1 - 1) ** 2 * (1 + (np.sin(3 * np.pi * x2)) ** 2)
        term3 = (x2 - 1) ** 2 * (1 + (np.sin(2 * np.pi * x2)) ** 2)

        y[i] = term1 + term2 + term3
    return y


@njit(parallel=True)
def rastr(X):
    """
    RASTRIGIN FUNCTION

    The Rastrigin function has several local minima.
    It is highly multimodal, but locations of the minima are regularly distributed.

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/rastr.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    if X.ndim<2:
        Xt = np.expand_dims(X, 0)
    else:
        Xt = X
    X = Xt
    d = X.shape[1]
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]

        sum = 0
        for ii in range(0, d):
            xi = xx[ii]
            sum = sum + (xi ** 2 - 10 * np.cos(2 * np.pi * xi))

        y[i] = 10 * d + sum
    return y


@njit(parallel=True)
def schaffer2(X):
    """
    SCHAFFER FUNCTION N. 2

    The second Schaffer function.

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/schaffer2.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    if X.ndim<2:
        Xt = np.expand_dims(X, 0)
    else:
        Xt = X
    X = Xt
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]

        x1 = xx[0]
        x2 = xx[1]

        fact1 = (np.sin(x1 ** 2 - x2 ** 2)) ** 2 - 0.5
        fact2 = (1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2

        y[i] = 0.5 + fact1 / fact2
    return y


@njit(parallel=True)
def schaffer4(X):
    """
    SCHAFFER FUNCTION N. 4

    The fourth Schaffer function.

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/schaffer4.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    if X.ndim<2:
        Xt = np.expand_dims(X, 0)
    else:
        Xt = X
    X = Xt
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]
        x1 = xx[0]
        x2 = xx[1]

        fact1 = (np.cos(np.sin(np.abs(x1 ** 2 - x2 ** 2)))) ** 2 - 0.5
        fact2 = (1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2

        y[i] = 0.5 + fact1 / fact2
    return y


@njit(parallel=True)
def schwef(X):
    """
    SCHWEFEL FUNCTION

    The Schwefel function is complex, with many local minima.

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/schwef.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    if X.ndim<2:
        Xt = np.expand_dims(X, 0)
    else:
        Xt = X
    X = Xt
    d = X.shape[1]
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]

        sum = 0
        for ii in range(0, d):
            xi = xx[ii]
            sum = sum + xi * np.sin(np.sqrt(np.abs(xi)))

        y[i] = 418.9829 * d - sum
    return y


@njit(parallel=True)
def shubert(X):
    """
    SHUBERT FUNCTION

    The Shubert function has several local minima and many global minima.

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/shubert.html
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Inputs:
    X = ndarray([[xx],[xx1],...])
    xx = ndarray([x1, x2, ..., xd])

    Parameters:

    Outputs:
    y = ndarray([f(xx), f(xx1), ....])
    """
    if X.ndim<2:
        Xt = np.expand_dims(X, 0)
    else:
        Xt = X
    X = Xt
    y = np.ones((X.shape[0],)) * np.inf
    for i in prange(X.shape[0]):
        xx = X[i, :]

        x1 = xx[0]
        x2 = xx[1]

        sum1 = 0
        sum2 = 0
        for ii in range(1, 5 + 1):
            new1 = ii * np.cos((ii + 1) * x1 + ii)
            new2 = ii * np.cos((ii + 1) * x2 + ii)
            sum1 = sum1 + new1
            sum2 = sum2 + new2

        y[i] = sum1 * sum2
    return y

