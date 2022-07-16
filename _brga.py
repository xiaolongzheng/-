import numpy as np
from utils import check_random_state
from evolutionary.gas._utils import rands, rws, elitism, sus, mpoints_mutation, mpoints_mutation_v, kpoints_crossover, rbs, uniform_crossover

class BRGA:
    """
    Binary-coded Real Genetic Algorithm

    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence or `Bounds`
        Bounds for variables.  There are two ways to specify the bounds:
        1. Instance of `Bounds` class.
        2. ``(min, max)`` pairs for each element in ``x``, defining the finite
        lower and upper bounds for the optimizing argument of `func`. It is
        required to have ``len(bounds) == len(x)``. ``len(bounds)`` is used
        to determine the number of parameters in ``x``.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : str, optional
        The differential evolution strategy to use. Should be one of:

            - 'best1bin'
            - 'best1exp'
            - 'rand1exp'
            - 'randtobest1exp'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'currenttobest1bin'
            - 'best2bin'
            - 'rand2bin'
            - 'rand1bin'

        The default is 'best1bin'

    maxiter : int, optional
        The maximum number of generations over which the entire POP is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * len(x)``
    popsize : int, optional
        A multiplier for setting the total POP size. The POP has
        ``popsize * len(x)`` individuals. This keyword is overridden if an
        initial POP is supplied via the `init` keyword. When using
        ``init='sobol'`` the POP size is calculated as the next power
        of 2 after ``popsize * len(x)``.
    tol : float, optional
        Relative tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(fitness))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by F.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        U[min, max). Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of POP
        stability.
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        Specify `seed` for repeatable minimizations.
    disp : bool, optional
        Prints the evaluated `func` at every iteration.
    callback : callable, `callback(xk, convergence=val)`, optional
        A function to follow the progress of the minimization. ``xk`` is
        the current value of ``x0``. ``val`` represents the fractional
        value of the POP convergence. When ``val`` is greater than one
        the function halts. If callback returns `True`, then the minimization
        is halted (any polishing is still carried out).
    polish : bool, optional
        If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`
        method is used to polish the best POP member at the end, which
        can improve the minimization slightly. If a constrained problem is
        being studied then the `trust-constr` method is used instead.
    maxfun : int, optional
        Set the maximum number of function evaluations. However, it probably
        makes more sense to set `maxiter` instead.
    init : str or array-like, optional
        Specify which type of POP initialization is performed. Should be
        one of:

            - 'latinhypercube'
            - 'sobol'
            - 'halton'
            - 'random'
            - array specifying the initial POP. The array should have
              shape ``(M, len(x))``, where M is the total POP size and
              len(x) is the number of parameters.
              `init` is clipped to `bounds` before use.

        The default is 'latinhypercube'. Latin Hypercube sampling tries to
        maximize coverage of the available parameter space.

        'sobol' and 'halton' are superior alternatives and maximize even more
        the parameter space. 'sobol' will enforce an initial POP
        size which is calculated as the next power of 2 after
        ``popsize * len(x)``. 'halton' has no requirements but is a bit less
        efficient. See `scipy.stats.qmc` for more details.

        'random' initializes the POP randomly - this has the drawback
        that clustering can occur, preventing the whole of parameter space
        being covered. Use of an array to specify a POP could be used,
        for example, to create a tight bunch of initial guesses in an location
        where the solution is known to exist, thereby reducing time for
        convergence.
    atol : float, optional
        Absolute tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(fitness))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    updating : {'immediate', 'deferred'}, optional
        If `immediate` the best solution vector is continuously updated within
        a single generation. This can lead to faster convergence as trial
        vectors can take advantage of continuous improvements in the best
        solution.
        With `deferred` the best solution vector is updated once per
        generation. Only `deferred` is compatible with parallelization, and the
        `workers` keyword can over-ride this option.
    workers : int or map-like callable, optional
        If `workers` is an int the POP is subdivided into `workers`
        sections and evaluated in parallel
        (uses `multiprocessing.Pool <multiprocessing>`).
        Supply `-1` to use all cores available to the Process.
        Alternatively supply a map-like callable, such as
        `multiprocessing.Pool.map` for evaluating the POP in parallel.
        This evaluation is carried out as ``workers(func, iterable)``.
        This option will override the `updating` keyword to
        `updating='deferred'` if `workers != 1`.
        Requires that `func` be pickleable.
    constraints : {NonLinearConstraint, LinearConstraint, Bounds}
        Constraints on the solver, over and above those applied by the `bounds`
        kwd. Uses the approach by Lampinen.
    x0 : None or array-like, optional
        Provides an initial guess to the minimization. Once the POP has
        been initialized this vector replaces the first (best) member. This
        replacement is done even if `init` is given an initial POP.
    """

    # mutation strategies
    _m_strategies = [""]

    def __init__(self,
                 func, constraints=(),  # 输入：适应度函数
                 convergence="iters", tol=0.01, atol=0, iterations=1000, polish=True, maxfun=np.inf,  # 输出：结束条件

                 seed=None,

                 # core alg params
                 init='latinhypercube', popsize=5, x0=None,  # 种群初始化参数
                 mutation='', UR=0.2,  # mutation strategy and setup params
                 CR=0.6,  # crossover probability setup

                 # running settup
                 callback=None, disp=True,
                 ):

        self.rnd = check_random_state(seed)

        # a wrapped function to reduce the func interface
        self.func = func

        # convert tuple of lower and upper bounds to limits
        # [(low_0, high_0), ..., (low_n, high_n] -> [[low_0, ..., low_n], [high_0, ..., high_n]]
        self.bounds = self.func.bounds
        self.boundlens = self.func.boundlens + 1
        self.nBits = self.func.nBits
        self.tDecs = np.array([int('1'*nb, 2) for nb in self.nBits])
        self.spacetype = self.func.spacetype
        self.encode_preci = self.boundlens/self.tDecs  # precision of encoding
        self.bin2decimal = np.frompyfunc(int, 2, 1)
        self.d2bins = np.frompyfunc(np.binary_repr, 1, 1)
        self.alian2Bits = np.frompyfunc(lambda x, bits: "0" * (bits - len(x))+x, 2, 1)

        # relative and absolute tolerances for convergence
        self.tol, self.atol = tol, atol
        if iterations is None:  # the default used to be None
            iterations = 1000
        self.iterations = iterations
        self.iter = 0  # 当前迭代次数
        if maxfun is None:  # the default used to be None
            maxfun = np.inf
        self.maxfun = maxfun
        self._nfev = 0
        self.polish = polish

        if convergence in ["iters"]:
            self.convergence_condition = self._reach_iterations
        elif convergence in ["accu"]:
            self.convergence_condition = self._reach_accuracy
        else:
            raise ValueError("Please select a valid convergence strategy")

        """initialize POP and fitness"""
        # POP is initialized with range [0, 1].
        self.dims = np.sum(self.nBits)
        self.popsize = popsize

        if init == 'latinhypercube':
            self.init_population_qmc(qmc_engine='latinhypercube')
        elif init == 'sobol':
            # must be Ns = 2**m for Sobol'
            n_s = int(2 ** np.ceil(np.log2(self.popsize)))
            self.popsize = n_s
            self.init_population_qmc(qmc_engine='sobol')

        elif init == 'halton':
            self.init_population_qmc(qmc_engine='halton')
        elif init == 'random':
            self.init_population_random()
        else:
            raise ValueError(init)
        self.init_method = init

        # prior experience (or knowledge, in other words, a guessed solution of the problem) for optimization problem
        if x0 is not None:
            x0_scaled = self._encode(np.asarray(x0))
            if ((x0_scaled > 1.0) | (x0_scaled < 0.0)).any():
                raise ValueError("entries of x0 lay outside the given bounds.")
            # inject the priors into POP
            self.POP[0:x0_scaled.shape[0]] = x0_scaled
        self.x0 = x0
        self.fitness = np.full(self.popsize, np.inf)  # POP fitness

        if mutation not in self._m_strategies:
            raise ValueError("Please select a valid mutation strategy")
        self.mutation = mutation

        # scale parameter for mutation
        self.UR = UR
        self.CR = CR  # crossover probability

        self.callback = callback

        # infrastructure for constraints
        self.constraints = constraints
        self.constraint_violation = np.zeros((self.popsize, 1))
        self.feasible = np.ones(self.popsize, bool)

        self.disp = disp

    def reinit(self):
        """initialize POP and fitness"""
        # POP is initialized with range [0, 1].
        init = self.init_method
        if init == 'latinhypercube':
            pass
        elif init == 'sobol':
            # must be Ns = 2**m for Sobol'
            n_s = int(2 ** np.ceil(np.log2(self.popsize)))
            self.popsize = n_s
            self.init_population_qmc(qmc_engine='sobol')
        elif init == 'halton':
            self.init_population_qmc(qmc_engine='halton')
        elif init == 'random':
            self.init_population_random()
        else:
            raise ValueError(init)

        # prior experience (or knowledge, in other words, a guessed solution of the problem) for optimization problem
        x0 = self.x0
        if x0 is not None:
            x0_scaled = self._encode(np.asarray(x0))
            if ((x0_scaled > 1.0) | (x0_scaled < 0.0)).any():
                raise ValueError("entries of x0 lay outside the given bounds.")
            # inject the priors into POP
            self.POP[0:x0_scaled.shape[0]] = x0_scaled

        self.fitness = np.full(self.popsize, np.inf)  # reset POP fitness
        self._nfev = 0
        self.iter = 0  # 当前迭代次数
        self.feasible = np.ones(self.popsize, bool)

    def init_population_qmc(self, qmc_engine):
        """Initializes the POP with a QMC method.

        QMC methods ensures that each parameter is uniformly
        sampled over its range.

        Parameters
        ----------
        qmc_engine : str
            The QMC method to use for initialization. Can be one of
            ``latinhypercube``, ``sobol`` or ``halton``.

        """
        from scipy.stats import qmc

        # Create an array for POP of candidate solutions.
        if qmc_engine == 'latinhypercube':
            sampler = qmc.LatinHypercube(d=self.dims, seed=self.rnd)
        elif qmc_engine == 'sobol':
            sampler = qmc.Sobol(d=self.dims, seed=self.rnd)
        elif qmc_engine == 'halton':
            sampler = qmc.Halton(d=self.dims, seed=self.rnd)
        else:
            raise ValueError(qmc_engine)

        self.POP = sampler.random(n=self.popsize)
        self.POP = np.rint(self.POP)
        self.POP = self.POP.astype(np.bool_)

    def init_population_random(self):
        """
        Initializes the POP at random.
        """
        self.POP = self.rnd.uniform(size=(self.popsize, self.dims))
        self.POP = np.rint(self.POP)
        self.POP = self.POP.astype(np.bool_)

    def init_population_array(self, init):
        """
        Initializes the POP with a user specified POP.

        Parameters
        ----------
        init : np.ndarray
            Array specifying subset of the initial POP. The array should
            have shape (M, len(x)), where len(x) is the number of parameters.
            The POP is clipped to the lower and upper bounds.
        """
        # make sure you're using a float array
        popn = np.asfarray(init)

        if (np.size(popn, 0) < 5 or
                popn.shape[1] != self.dims or
                len(popn.shape) != 2):
            raise ValueError("The POP supplied needs to have shape"
                             " (M, len(x)), where M > 4.")

        # scale values and clip to bounds, assigning to POP
        self.POP = self._encode(popn)

    @property
    def x(self):
        """
        The best solution from the solver
        """
        return self._decode(np.expand_dims(self.fbest, axis=0))

    def _reach_accuracy(self):
        """
        Return True if the solver has converged.
        """
        if np.any(np.isinf(self.fitness)):
            return False

        return (np.std(self.fitness) <=
                self.atol +
                self.tol * np.abs(np.mean(self.fitness)))

    def _reach_iterations(self):
        """
        reach the max iterations
        """
        return self.iter > self.iterations

    def solve(self):
        """
        Runs the DifferentialEvolution.

        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.  If `polish`
            was employed, and a lower minimum was obtained by the polishing,
            then OptimizeResult also contains the ``jac`` attribute.
        """

        # The POP may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies.
        # Although this is also done in the evolve generator it's possible
        # that someone can set maxiter=0, at which point we still want the
        # initial energies to be calculated (the following loop isn't run).
        if np.all(np.isinf(self.fitness)):
            self.param_POP = self._decode(self.POP)
            self.feasible, self.constraint_violation = self._calculate_feasibilities(self.param_POP)

            # only work out POP fitness for feasible solutions
            self.fitness[self.feasible] = self._calculate_fitnesses(self.param_POP[self.feasible])

            # find the best feasible solution
            ibest = np.argmin(self.fitness)
            self.fbest = self.POP[ibest]
            self.bfitness = self.fitness[ibest]

        # do the optimization.
        for nit in range(1, self.iterations + 1):
            # evolve the POP by a generation
            self.iter = nit

            # parent selection strategy
            mates = rands(self.popsize, self.rnd)
            # mates = sus(self.fitness, self.popsize, self.rnd)
            # mates = rws(self.fitness, self.popsize, self.rnd)
            # mates = rbs(self.fitness, self.popsize, self.rnd)

            # crossover and mutation strategy
            offspring = np.ones_like(self.POP)*np.inf
            for p in range(0, self.popsize, 2):
                Child1, Child2 = kpoints_crossover(self.POP[mates[p]], self.POP[mates[p+1]], self.CR, self.rnd, k=50)
                # Child1, Child2 = uniform_crossover(self.POP[mates[p]], self.POP[mates[p+1]], self.CR, self.rnd)
                # Child1 = mpoints_mutation(Child1, self.UR, self.rnd)
                # Child2 = mpoints_mutation(Child2, self.UR, self.rnd)
                Child1 = mpoints_mutation_v(Child1, self.UR, self.rnd.rand(self.dims))
                Child2 = mpoints_mutation_v(Child2, self.UR, self.rnd.rand(self.dims))

                offspring[p] = Child1
                offspring[p+1] = Child2

            offspring = offspring.astype(np.bool_)

            # bound constrain
            pass

            # evaluate POP for solutions
            self.param_off = self._decode(offspring)
            self.feasible, self.constraint_violation = self._calculate_feasibilities(self.param_off)
            off_fitness = np.full_like(self.fitness, np.inf)
            off_fitness[self.feasible] = self._calculate_fitnesses(self.param_off[self.feasible])

            # selection
            com_pop = np.vstack((self.POP, offspring))
            com_fitness = np.hstack((self.fitness, off_fitness))

            # find the best feasible solution
            ibest = np.argmin(off_fitness)
            if self.bfitness > off_fitness[ibest]:
                self.fbest = offspring[ibest]
                self.bfitness = off_fitness[ibest]

            # mask = elitism(com_fitness, self.popsize, self.rnd)
            mask = sus(com_fitness, self.popsize, self.rnd)
            # mask = rws(com_fitness, self.popsize, self.rnd)
            # mask = rbs(com_fitness, self.popsize, self.rnd)
            self.POP = com_pop[mask]
            self.fitness = com_fitness[mask]

            # self.POP = offspring
            # self.fitness = off_fitness
            # self.POP[0] = self.fbest
            # self.fitness[0] = self.bfitness


            if self.disp:
                print("realcoded_ga step %d: f(x)= %g"
                      % (self.iter, self.bfitness))

            if self.callback:
                pass

            # should the solver terminate?
            if self.convergence_condition():
                break

        return self.fbest, self.bfitness

    def _calculate_fitnesses(self, POP):
        """
        Calculate the energies of a POP.

        Parameters
        ----------
        POP : ndarray
            An array of parameter vectors normalised to [0, 1] using lower
            and upper limits. Has shape ``(np.size(POP, 0), len(x))``.

        Returns
        -------
        fitness : ndarray
            An array of energies corresponding to each POP member. If
            maxfun will be exceeded during this call, then the number of
            function evaluations will be reduced and energies will be
            right-padded with np.inf. Has shape ``(np.size(POP, 0),)``
        """
        num_members = POP.shape[0]
        fitness = self.func(POP)
        self._nfev += num_members
        return fitness

    def _calculate_feasibilities(self, POP):
        """
        Calculate the feasibilities of a POP.

        Parameters
        ----------
        POP : ndarray
            An array of parameter vectors normalised to [0, 1] using lower
            and upper limits. Has shape ``(np.size(POP, 0), len(x))``.

        Returns
        -------
        feasible, constraint_violation : ndarray, ndarray
            Boolean array of feasibility for each POP member, and an
            array of the constraint violation for each POP member.
            constraint_violation has shape ``(np.size(POP, 0), M)``,
            where M is the number of constraints.
        """
        feasible = np.ones(POP.shape[0], bool)
        constraint_violation = np.zeros((POP.shape[0], 1))
        if not self.constraints:
            return feasible, constraint_violation

        constraint_violation = np.array([c.violations(POP) for c in self.constraints])

        feasible = ~(np.sum(constraint_violation, axis=0) > 0)

        return feasible, constraint_violation

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return

    def __next__(self):
        """
        Evolve the POP by a single generation

        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        self.iter += 1

        CR, F = self.randFCR(self.popsize, 0.4, 0.1, 0.3, 0.1)
        CR = np.tile(CR, self.dims)
        F = np.tile(F, self.dims)

        # CR = np.ones_like(self.POP) * self.CR
        # F = np.ones_like(self.POP) * self.F  # np.tile(a, 2)
        trails = self.get_trails(m_strategy=self.mutation, POP=self.POP, F=F, CR=CR)

        # evaluate POP for solutions
        self.param_trails = self._decode(trails)
        self.feasible, self.constraint_violation = self._calculate_feasibilities(self.param_trails)
        trail_fitness = np.full_like(self.fitness, np.inf)
        trail_fitness[self.feasible] = self._calculate_fitnesses(self.param_trails[self.feasible])
        mask = self.fitness > trail_fitness
        self.POP[mask] = trails[mask]
        self.fitness[mask] = trail_fitness[mask]
        # print(trail_fitness)

        # find the best feasible solution
        ibest = np.argmin(self.fitness)
        if self.bfitness > self.fitness[ibest]:
            self.fbest = self.POP[ibest]
            self.bfitness = self.fitness[ibest]

        if self.disp:
            print("differential_evolution step %d: f(x)= %g"
                  % (self.iter, self.bfitness))

        if self.callback:
            pass

        return self.fbest, self.bfitness

    def bins2Decimals(self, X, params):
        X = X.astype(np.uint0)
        i_start = 0
        bins_sections = []
        for section in params:
            t = np.apply_along_axis("".join, 1, X[:, i_start:i_start+section].astype(str))
            bins_sections.append(self.bin2decimal(t, 2))
            i_start += section
        Xdecs = np.array(bins_sections).T
        return Xdecs

    def _decode(self, X):
        """
        decode X from the search space.
        :param X: ndarray
        :return:
        """
        Xdecs = self.bins2Decimals(X, self.nBits)
        # print(Xdecs)
        rX = self.bounds[0] + Xdecs/self.tDecs * self.boundlens
        # print(rX)
        if self.spacetype == "discrete":
            return rX.astype(int)
        elif self.spacetype == "binary":
            return np.rint(rX)
        rX = rX.astype(np.float64)
        return rX

    def alianBits(self, X, params):
        bins_sections = []
        for d, bits in enumerate(params):
            t = self.alian2Bits(X[:,d], bits)
            bins_sections.append(t)
        Xdecs = np.array(bins_sections).T
        return Xdecs

    def _encode(self, X):
        """
        Encode X into the search space.
        :param X: ndarray
        :return:
        """
        Xdecs = ((X - self.bounds[0])/self.boundlens*self.tDecs).astype(np.uint64)
        # print(Xdecs)
        Xbins = self.d2bins(Xdecs)
        Xbins = self.alianBits(Xbins, self.nBits)
        # print(Xbins)
        bX = np.apply_along_axis("".join, 1, Xbins).reshape((Xbins.shape[0], 1))
        Xs=np.apply_along_axis(lambda x: np.array(list(x[0])).astype(np.bool_), 1, bX)
        return Xs


if __name__ == '__main__':
    from benchmarks import manyf_settings as f_settings, ackley

    symbol = "ackley"
    settings = f_settings(dims=2)
    # bounds = settings[symbol]["bounds"]
    bounds = [(-32, 32)]
    from utils import _FunctionWrapper

    alg = BRGA(func=_FunctionWrapper(eval(symbol), bounds, nbits=[12, 12], spacetype="discrete"), popsize=50, iterations=200)
    # rX = alg.encode(np.array([[-32, -31],[-30, -16]]), alg.nBits)
    # rX = alg.encode(np.array([[32, 31], [0, 1]]), alg.nBits)
    # bX = alg.decode(rX, alg.nBits)
    # print(rX.astype(np.uint0))
    # print(bX)
    # print(alg.encode_preci)

    alg.solve()
    print(alg.x)
    print(settings[symbol]["x*"])
    print(alg.bfitness)
    print(settings[symbol]["fmin"])

    # from evolutionary.de.benchmarks.constrained import constrainedf_settings, PrG1f
    #
    # symbol = "PrG1f"
    # settings = constrainedf_settings()
    # bounds = settings[symbol]["bounds"]
    # from evolutionary.de.utils import _FunctionWrapper, _ConstraintWrapper
    #
    # alg = DifferentialEvolution(func=_FunctionWrapper(eval(symbol), bounds), constraints=(_ConstraintWrapper(settings[symbol]["cfs"], settings[symbol]["cfs"])))
    # alg.solve()
    # print(alg.x, alg.bfitness)
