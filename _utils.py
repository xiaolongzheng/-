import numpy as np


def sus(FitnV, Nsel, rnd, ibest="small"):
    """
    Stochastic Universal Sampling,
    to select Nsel individuals accroding to their fitness FitnV.
    The best fitness value, the more selection probabilities.
    :param FitnV: fitness values of population
    :param Nsel: number of individuals to be selected
    :param rnd: random generator
    :param ibest: "small": smaller fitness is the best, "big":the bigger one is the best
    :return:
    """
    # Identify the population size(Nind)
    Nind = FitnV.size
    if ibest == "small":
        FitnV = np.max(FitnV) - FitnV

    # Perform stochastic universal sampling
    cumfit = np.cumsum(FitnV)
    trials = (cumfit[-1] / Nsel) * (rnd.rand(1, ) + np.arange(0, Nsel))
    Mf = np.tile(cumfit, (Nsel, 1)).T
    Mt = np.tile(trials, (Nind, 1))
    Mcom = (Mt < Mf) & (np.vstack((np.zeros((1, Nsel)), Mf[:-1, :])) <= Mt)
    NewChrIx = np.array(Mcom.nonzero()[0])
    rnd.shuffle(NewChrIx)

    return NewChrIx


def rws(FitnV, Nsel, rnd, ibest="small"):
    """
    Roulette Wheel Selection
    to select Nsel individuals accroding to their fitness FitnV.
    The greater fitness value, the more selection probabilities.
    :param FitnV: fitness values of population
    :param Nsel: number of individuals to be selected
    :param rnd: random generator
    :param ibest: "small": smaller fitness is the best, "big":the bigger one is the best
    :return:
    """
    # Identify the population size(Nind)
    Nind = FitnV.size
    if ibest == "small":
        FitnV = np.max(FitnV) - FitnV

    # Perform stochastic universal sampling
    cumfit = np.cumsum(FitnV)
    trials = cumfit[-1] * rnd.rand(Nsel, )
    Mf = np.tile(cumfit, (Nsel, 1)).T
    Mt = np.tile(trials, (Nind, 1))
    Mcom = (Mt < Mf) & (np.vstack((np.zeros((1, Nsel)), Mf[:-1, :])) <= Mt)
    NewChrIx = np.array(Mcom.nonzero()[0])
    rnd.shuffle(NewChrIx)

    return NewChrIx


def rbs(FitnV, Nsel, rnd, k=3, strategy=0, ibest="small"):
    """
    Tournament Selection
    to select Nsel individuals accroding to their fitness FitnV.
    The greater fitness value, the more selection probabilities.
    :param FitnV: fitness values of population
    :param Nsel: number of individuals to be selected
    :param rnd: random generator
    :param ibest: "small": smaller fitness is the best, "big":the bigger one is the best
    :param k: competitors in each run
    :return:
    """
    # Identify the population size(Nind)
    Nind = FitnV.size
    if ibest == "small":
        FitnV = np.max(FitnV) - FitnV

    indexs = np.arange(0, Nind)

    NewChrIx = np.ones((Nsel,), dtype=np.int32)
    for i in range(Nsel):
        around = rnd.choice(indexs, k)
        if strategy == 0:
            t = np.argmax(FitnV[around])
            NewChrIx[i] = around[t]
        elif strategy == 1:
            NewChrIx = 0
            pass  # iteratively selection
            # for i in list(np.arange(0, k, step=2)):
            #     if

    # rnd.shuffle(NewChrIx)

    return NewChrIx


def rands(Nind, rnd):
    """
    random mating
    mate individual randomly
    :param Nind: population size
    :param rnd: random generator
    :return:
    """
    indexs = np.arange(0, Nind)
    NewChrIx = rnd.permutation(indexs)
    return NewChrIx


def elitism(FitnV, Nsel, rnd, ibest="small"):
    """
    Elitism Selection
    to select Nsel individuals accroding to their fitness FitnV.
    The greater fitness value, the more selection probabilities.
    :param FitnV: fitness values of population
    :param Nsel: number of individuals to be selected
    :param rnd: random generator
    :param ibest: "small": smaller fitness is the best, "big":the bigger one is the best
    :return:
    """
    # Identify the population size(Nind)
    Nind = FitnV.size
    if Nsel > Nind:
        raise Exception("intent to select Nsel individuals when their are only Nind(<Nsel) individuals.")

    if ibest == "small":
        FitnV = np.max(FitnV) - FitnV

    sInds = np.argsort(FitnV)
    NewChrIx = sInds[-Nsel:]

    rnd.shuffle(NewChrIx)

    return NewChrIx


def sbc(Parent1, Parent2, eta, rnd):
    """
    SBC Simulate Binary Crossover.
    :param Parent1: ndarray, parent for crossover.
    :param Parent2: ndarray, parent for crossover.
    :param eta: distribution index.
    :return:
    """

    dim = Parent1.size

    u = rnd.rand(1, dim)
    cf = np.zeros((1, dim))
    mask = u <= 0.5
    cf[mask] = np.power((2 * u[mask]), 1 / (eta + 1))
    mask = u > 0.5
    cf[mask] = np.power((2 * (1 - u[mask])), -1 / (eta + 1))
    Child1 = 0.5 * ((1 + cf) * Parent1 + (1 - cf) * Parent2)
    Child2 = 0.5 * ((1 - cf) * Parent1 + (1 + cf) * Parent2)
    return Child1, Child2


def polynomial_mutation(Individual, bu, bd, eta_m, rnd):
    """
    Polynomial Mutation
    :param Individual: ndarray, [dim1,dim2,dim3,...,dim k]
    :param bu: upbound of every dim
    :param bd: lowbound of every dim
    :param eta_m: distribution index
    :return:
    """
    Individual = Individual.squeeze()
    dim = Individual.size
    nIndividual = []
    for j in range(0, dim):
        y = Individual[j]
        yd = bd[j]
        yu = bu[j]
        if y > yd:
            if (y - yd) < (yu - y):
                delta = (y - yd) / (yu - yd)
            else:
                delta = (yu - y) / (yu - yd)
            r2 = rnd.rand(1, )
            indi = 1 / (eta_m + 1)
            if r2 <= 0.5:
                xy = 1 - delta
                val = 2 * r2 + (1 - 2 * r2) * (xy ** (eta_m + 1))
                deltaq = val ** indi - 1
            else:
                xy = 1 - delta
                val = 2 * (1 - r2) + 2 * (r2 - 0.5) * (xy ** (eta_m + 1))
                deltaq = 1 - val ** indi
            y = y + deltaq * (yu - yd)
            y = min([y, yu])
            y = max([y, yd])
            nIndividual.append(y)
        else:  # y <= yd
            nIndividual.append(rnd.rand(1, ) * (yu - yd) + yd)
    nIndividual = np.array(nIndividual, dtype=np.float64).squeeze()
    return nIndividual


def kpoints_crossover(Parent1, Parent2, pc, rnd, k=2):
    nBits = Parent1.size
    if k > nBits / 2:
        raise Exception("k is not allowed to be bigger than nBits/2.")
    Child1 = Parent1.copy()
    Child2 = Parent2.copy()
    Points = np.sort(rnd.randint(nBits, size=2 * k))
    for i in range(0, k):
        if rnd.rand(1) < pc:
            Child1[Points[2 * i]:Points[2 * i + 1]] = Parent2[Points[2 * i]:Points[2 * i + 1]]
            Child2[Points[2 * i]:Points[2 * i + 1]] = Parent1[Points[2 * i]:Points[2 * i + 1]]
            # Child1[Points[2 * i]:Points[2 * i + 1]+1] = Parent2[Points[2 * i]:Points[2 * i + 1]+1]
            # Child2[Points[2 * i]:Points[2 * i + 1]+1] = Parent1[Points[2 * i]:Points[2 * i + 1]+1]
    return Child1, Child2


def uniform_crossover(Parent1, Parent2, pc, rnd):
    nBits = Parent1.size
    Child1 = Parent1.copy()
    Child2 = Parent2.copy()
    for i in range(0, nBits):
        if rnd.rand(1) < pc:
            Child1[i] = Parent2[i]
            Child2[i] = Parent1[i]
    return Child1, Child2


def ordered_crossover(Parent1, Parent2, pc, rnd, k=2):
    """
    for combination optimization
    :param Parent1: ndarray
    :param Parent2: ndarray
    :param pc:
    :param rnd: random generator
    :param k:
    :return:
    """
    nBits = Parent1.size
    Child1 = np.ones_like(Parent1) * np.inf
    Child2 = np.ones_like(Parent2) * np.inf
    Points = np.sort(rnd.randint(nBits, size=2 * k))
    for i in range(0, k):
        if rnd.rand(1) < pc:
            Child1[Points[2 * i]:Points[2 * i + 1]] = Parent2[Points[2 * i]:Points[2 * i + 1]]
            Child2[Points[2 * i]:Points[2 * i + 1]] = Parent1[Points[2 * i]:Points[2 * i + 1]]
    orders = [x for x in range(nBits)]
    orders = orders[Points[-1]:] + orders[:Points[-1]]
    for c in orders:
        if any(np.isinf(Child1)):
            bits = set(Child1)
            for bit in orders:
                if Parent1[bit] not in bits:
                    Child1[c] = Parent1[bit]
                    break
        else:
            break
    for c in orders:
        if any(np.isinf(Child2)):
            bits = set(Child2)
            for bit in orders:
                if Parent2[bit] not in bits:
                    Child2[c] = Parent2[bit]
                    break
        else:
            break
    return Child1, Child2


from numba import njit, prange


@njit(parallel=True)
def mpoints_mutation_v(Individual, pm, rnds):
    """
    multi-point Mutation
    :param Individual: ndarray, [dim1,dim2,dim3,...,dim k]
    :param pm: mutation rate
    :param rnd: random generator
    :return:
    """
    nBits = Individual.size
    nIndividual = Individual.copy()
    for i in prange(0, nBits):
        if rnds[i] < pm:
            nIndividual[i] = not Individual[i]

    return nIndividual


def mpoints_mutation(Individual, pm, rnd):
    """
    multi-point Mutation
    :param Individual: ndarray, [dim1,dim2,dim3,...,dim k]
    :param pm: mutation rate
    :param rnd: random generator
    :return:
    """
    nBits = Individual.size
    nIndividual = Individual.copy()
    for i in range(0, nBits):
        if rnd.rand(1) < pm:
            nIndividual[i] = not Individual[i]

    return nIndividual


if __name__ == '__main__':
    ordered_crossover(np.array([1, 2, 3, 4, 5, 6]), np.array([3, 4, 5, 1, 2, 6]), 1.9, np.random)
    # rands(30, np.random)
    # uniform_crossover(np.array([1, 2, 3, 4, 5, 6]), np.array([11, 21, 31, 41, 51, 61]), 0.9, np.random)
    # mpoints_mutation(np.array([1, 2, 3, 4, 5, 6]), 3, np.random)
    # rws(np.array([1,2,3,4,5,6]), 3, np.random)
    # elitism(np.array([1, 2, 4, 3, 5, 6]), 3, np.random)
    # sbc(np.array([1, 2, 3]), np.array([5, 6, 7]), 9, np.random)
    # polynomial_mutation(np.array([1, 2, 3]), np.array([10, 20, 30]), np.array([0, 0, 0]), 5, np.random)
