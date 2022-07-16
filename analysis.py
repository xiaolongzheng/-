import time
from benchmarks import crossit, manyf_settings as f_settings
# from benchmarks import spheref, bowlf_settings as f_settings

symbol = "crossit"
settings = f_settings(dims=5)
bounds = settings[symbol]["bounds"]
from benchmarks.utils import _FunctionWrapper, differentialGroupring

allgroups, epsilons = differentialGroupring(func=_FunctionWrapper(eval(symbol), bounds))
print(allgroups)
print(epsilons)