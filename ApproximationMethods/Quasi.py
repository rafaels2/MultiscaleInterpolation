import numpy as np
from cachetools import cached
from Tools.Utils import generate_kernel, generate_cache

from .ApproximationMethod import ApproximationMethod
from Tools.SamplingPoints import SamplingPointsCollection


def combine(a, b):
    def func(x, y):
        return a(x, y), b(x, y)

    return func


class Quasi(ApproximationMethod):
    def __init__(self, manifold, original_function, grid_parameters, rbf,
                 scale, is_approximating_on_tangent):
        self._values_to_average_counter = []
        if isinstance(original_function, tuple):
            original_function = combine(*original_function)
            self._is_adaptive = True
        else:
            self._is_adaptive = False
        super().__init__(manifold, original_function, grid_parameters, rbf)
        self._is_approximating_on_tangent = is_approximating_on_tangent
        self._rbf_radius = scale

        self._grid = SamplingPointsCollection(self._rbf_radius,
                                              original_function,
                                              grid_parameters,
                                              phi_generator=self._calculate_phi)

        self._kernel = generate_kernel(self._rbf, self._rbf_radius)

    def average_support_size(self):
        return np.average(np.array(self._values_to_average_counter))

    @staticmethod
    def _get_weights_for_point(point, x, y):
        return point.phi(x, y)

    def _get_values_to_average(self, x, y):
        values_to_average = list()
        weights = list()

        for point in self._grid.points_in_radius(x, y):
            values_to_average.append(point.evaluation)
            weights.append(self._get_weights_for_point(point, x, y))

        return values_to_average, weights

    @staticmethod
    def _normalize_weights(weights):
        normalizer = sum(weights)

        if normalizer == 0:
            normalizer = 0.00001

        return [w_i / normalizer for w_i in weights]

    @cached(cache=generate_cache(maxsize=1000))
    def approximation(self, x, y):
        """ Average sampled points around (x, y), using phis as weights """
        values_to_average, weights = self._get_values_to_average(x, y)
        weights = self._normalize_weights(weights)

        if self._is_approximating_on_tangent:
            return sum(w_i * x_i for w_i, x_i in zip(weights, values_to_average))

        self._values_to_average_counter.append(len(weights))
        return self._manifold.average(values_to_average, weights)
