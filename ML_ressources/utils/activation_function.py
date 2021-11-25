# collection of activation functions

import numpy as np


def choose_activation(func_name):
    if func_name == "identity":
        return Identity()
    elif func_name == "tanh":
        return TanH()
    else:
        raise NotImplementedError


class Identity():
    # Identity: range (-inf. , inf.)
    def __call__(self, x):
        return x

    def gradient(self, x):
        return 1


class TanH():
    # Hyperbolic tangent: range (-1, 1)
    def __call__(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)
