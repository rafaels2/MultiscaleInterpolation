import numpy as np
from cachetools import cached

from Tools.SamplingPoints import SamplingPointsCollection
from Tools.Utils import generate_kernel, generate_cache
from .ApproximationMethod import ApproximationMethod


def combine(a, b):
    def func(x, y):
        return a(x, y), b(x, y)
    return func


class Quasi(ApproximationMethod):
    def __init__(self, manifold, original_function, confidence, grid_parameters, rbf,
                 rbf_radius, is_approximating_on_tangent):
        if isinstance(original_function, tuple):
            original_function = combine(*original_function)
            self._is_adaptive = True
        else:
            self._is_adaptive = False
        super().__init__(manifold, original_function, grid_parameters, rbf)
        self._is_approximating_on_tangent = is_approximating_on_tangent

        self._confidence = confidence
        self._rbf_radius = rbf_radius
        self._grid = SamplingPointsCollection(rbf_radius,
                                              original_function,
                                              confidence,
                                              grid_parameters,
                                              phi_generator=self._calculate_phi)

        self._kernel = generate_kernel(self._rbf, rbf_radius)

    def _calculate_phi(self, x_0, y_0):
        point = np.array([x_0, y_0])

        @cached(cache=generate_cache(maxsize=100))
        def phi(x, y):
            vector = np.array([x, y])
            return self._kernel(vector, point)

        return phi
    
    @cached(cache=generate_cache(maxsize=1000))
    def approximation(self, x, y):
        """ Average sampled points around (x, y), using phis as weights """
        values_to_average = list()
        weights = list()
        base = None

        if self._is_adaptive:
            base = self._original_function(x, y)[1]

        for point in self._grid.points_in_radius(x, y):
            if self._is_adaptive:
                values_to_average.append(self._manifold.exp(base, point.evaluation[0]))
            else:
                values_to_average.append(point.evaluation)
            weights.append(point.phi(x, y))

        normalizer = sum(weights)

        if normalizer == 0:
            normalizer = 0.00001

        weights = [w_i / normalizer for w_i in weights]
        if self._is_approximating_on_tangent:
            return sum(w_i * x_i for w_i, x_i in zip(weights, values_to_average))

        if self._is_adaptive:
            return self._manifold.log(base, self._manifold.average(values_to_average, weights))
        
        return self._manifold.average(values_to_average, weights)
