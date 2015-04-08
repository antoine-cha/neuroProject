#import numpy as np


class GradientAscent:
    def __init__(self, func, grad, x0):
        self.func = func
        self.grad = grad
        self.x0 = x0

    def iterate(self, x, step=1.0):
        """
        Operate one iteration of gradient ascent
        ----------------------------------------
        x : current point
        """
        return x + step * self.grad(x)

    def batch_iterate(self, x, step=1.0)
        """
        Operate one iteration of stochastic gradient ascent
        ----------------------------------------
        x : n_examples * 20 * 20 tensor
            batch of examples
        """
        return 0

