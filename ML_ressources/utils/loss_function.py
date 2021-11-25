# collection of loss functions

import numpy as np


def choose_loss(func_name):
    if func_name == "square loss":
        return SquareLoss()
    else:
        raise NotImplementedError


class SquareLoss():
    def __call__(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return - (y - y_pred)