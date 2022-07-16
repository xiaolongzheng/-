from evolutionary.de.benchmarks.others import *
from evolutionary.de.utils import _FunctionWrapper

if __name__ == '__main__':
    symbol = "stybtang"
    func = _FunctionWrapper(eval(symbol), settings[symbol]["bounds"])
    X = np.array(settings[symbol]["x*"])
    # X = np.array([[2.20, 1.57]])
    print(X)
    print(func(X).reshape(X.shape[0],1))
    print(settings[symbol]["fmin"])