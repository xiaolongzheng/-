import numpy as np


class _FunctionWrapper:
    """
    wrap user's cost function.
    for a function f(x, a, b, c), it can be wrapper as f, args=(a_value, b_value, c_value)
    """

    def __init__(self, f, bounds=[], args=(), nbits=[], spacetype="discrete"):
        self.f = f
        self.args = [] if args is None else args
        self.bounds = np.array(bounds, dtype='float').T
        self.boundlens = self.bounds[1] - self.bounds[0]
        self.dims = self.bounds.shape[1]
        self.nBits = np.array(nbits, dtype=np.int64)  # how many bits for each dim
        self.spacetype = spacetype  # the type of search space, value:'binary', 'discrete', 'continuous'
        # for the purpose of compiling the function by numba
        X = self.bounds[0].reshape(1, self.bounds.shape[1])
        self.f(X, *self.args)

    def __call__(self, x):
        return self.f(x, *self.args)


class _ConstraintWrapper:
    """wrap user's defined constraints.

    Parameters
    ----------
    wrap_func : wrapped constraint function.
    flb: lower bound of the func
    fub: upper bound of the func
    args: params for the func,
    for a function f(x, a, b, c), it can be wrapper as func, args=(a_value, b_value, c_value)
    Attributes
    ----------
    fun : callable
        Function defining the constraint wrapped by one of the convenience
        classes.
    """

    def __init__(self, func, flb=-np.inf, fub=np.inf, args=()):
        self.f = func
        self.args = args
        self.flb = flb
        self.fub = fub

    def __call__(self, x):
        return np.atleast_1d(self.f(x, *self.args))

    def violation(self, x):
        """How much the constraint is exceeded by.

        Parameters
        ----------
        x : array-like
            Vector of independent variables

        Returns
        -------
        excess : array-like
            How much the constraint is exceeded by
        """
        ev = self.f(np.asarray(x), *self.args)

        excess_lb = np.maximum(self.flb - ev, 0)
        excess_ub = np.maximum(ev - self.fub, 0)

        return excess_lb + excess_ub

    def violations(self, X):
        """How much the constraint is exceeded by.

        Parameters
        ----------
        X : array-like
            Vectors of independent variables

        Returns
        -------
        excess : array-like
            How much the constraint is exceeded by
        """
        EV = self.f(X, *self.args)

        excess_lb = np.maximum(self.flb - EV, 0)
        excess_ub = np.maximum(EV - self.fub, 0)

        return excess_lb + excess_ub


class _ConstraintWrappers:
    """wrap user's defined constraints.

    Parameters
    ----------
    wrap_func : wrapped constraint function.
    flb: lower bound of the func
    fub: upper bound of the func
    args: params for the func,
    for a function f(x, a, b, c), it can be wrapper as func, args=(a_value, b_value, c_value)
    Attributes
    ----------
    fun : callable
        Function defining the constraint wrapped by one of the convenience
        classes.
    """

    def __init__(self, funcs, fbs, args=()):
        self.f = funcs
        self.args = args
        self.fbounds = np.array(fbs, dtype='float').T
        self.flb = self.fbounds[0]
        self.fub = self.fbounds[1]

    def __call__(self, x):
        return np.atleast_1d(self.f(x, *self.args))

    def violation(self, x):
        """How much the constraint is exceeded by.

        Parameters
        ----------
        x : array-like
            Vector of independent variables

        Returns
        -------
        excess : array-like
            How much the constraint is exceeded by
        """
        ev = self.f(np.asarray(x), *self.args)

        excess_lb = np.maximum(self.flb - ev, 0)
        excess_ub = np.maximum(ev - self.fub, 0)

        return excess_lb + excess_ub

    def violations(self, X):
        """How much the constraint is exceeded by.

        Parameters
        ----------
        X : array-like
            Vectors of independent variables

        Returns
        -------
        excess : array-like
            How much the constraint is exceeded by
        """
        EV = self.f(X, *self.args)

        excess_lb = np.maximum(self.flb - EV, 0)
        excess_ub = np.maximum(EV - self.fub, 0)

        return excess_lb + excess_ub


def differentialGroupring(func, epsilon=0.1):
    """
    Decomposition Strategies: Differential Grouping Algorithm
    ref: Omidvar, Mohammad Nabi, et al.
    "Cooperative co-evolution with differential grouping for large scale optimization."
    IEEE Transactions on evolutionary computation 18.3 (2013): 378-393.
    :param func: a wrapped function
    :param epsilon: The choice of epsilon affects the sensitivity of the algorithm in detecting the interactions
    between the variables.
    A smaller epsilon makes the algorithm more sensitive to very weak interactions between the decision variables.
    :return:
    """
    dims = func.dims
    visited = np.zeros((dims,), dtype=np.bool_)
    epsilons = np.zeros((dims, dims))
    seps = []
    allgroups = []
    for i in np.arange(dims):
        if visited[i]:
            continue
        else:
            visited[i] = True

        group = [i]
        for j in np.arange(i + 1, dims):
            p1 = func.bounds[0].copy()
            p2 = p1.copy()
            p2[i] = (np.random.rand(1) * 0.3 + 0.5) * func.boundlens[i] + p2[i]
            delta1 = func(np.expand_dims(p1, axis=0)) - func(np.expand_dims(p2, axis=0))
            p1[j] = (np.random.rand(1) * 0.3 + 0.5) * func.boundlens[j] + p1[j]
            p2[j] = p1[j]
            delta2 = func(np.expand_dims(p1, axis=0)) - func(np.expand_dims(p2, axis=0))
            epsilons[i, j] = np.abs(delta1 - delta2)
            if not visited[j] and np.abs(delta1 - delta2) > epsilon:
                group.append(j)
                visited[j] = True
        if len(group) == 1:
            seps.append(group)
        else:
            allgroups.append(group)
    allgroups += seps
    return allgroups, epsilons
