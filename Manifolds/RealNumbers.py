import numpy as np

from .AbstractManifold import AbstractManifold
from . import register_manifold


@register_manifold("numbers")
class RealNumbers(AbstractManifold):
    def exp(self, x, y):
        return x + y

    def log(self, x, y):
        return y - x

    def _to_numbers(self, x):
        return x

    def zero_func(self, x_0, x_1):
        return 2

    def _get_geodetic_line(self, x, y):
        def line(t):
            return x + (y - x) * (1 - t)

        return line


@register_manifold("no_norm")
class NoNormalizationNumbers(RealNumbers):
    def average(self, values_to_average, weights):
        return sum([w_i * v_i for w_i, v_i in zip(weights, values_to_average)])

    def calculate_error(self, x, y):
        """ Relative Error """
        error = np.zeros_like(x, dtype=np.float32)
        for index in np.ndindex(x.shape):
            error[index] = self.distance(x[index], y[index])
        return error


@register_manifold("no_norm_calibration")
class Calibration(NoNormalizationNumbers):
    def calculate_error(self, x, y):
        error = np.zeros_like(x, dtype=np.float32)
        for index in np.ndindex(x.shape):
            error[index] = x[index] / y[index]
        return error


class PositiveNumbers(RealNumbers):
    def exp(self, x, y):
        return x ** y

    def log(self, x, y):
        # TODO: Think if we want to change to log(1+x)
        if x == 1:
            epsilon = 0.00001
        else:
            epsilon = 0
        return np.log(y) / (np.log(x) + epsilon)
