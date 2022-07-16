from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec



def target(x):
    return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)

optimizer = BayesianOptimization(target, {'x': (-2, 10)}, random_state=27)
optimizer.maximize(init_points=2, n_iter=5, kappa=5)