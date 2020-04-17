import numpy as np
from cachetools import cached
from numpy import linalg as la
from collections import namedtuple
from Tools.Utils import generate_grid, generate_kernel, evaluate_on_grid, generate_cache

from .ApproximationMethod import ApproximationMethod
from Tools.SamplingPoints import SamplingPointsCollection


class Quasi(ApproximationMethod):
    def __init__(self, manifold, original_function, grids_parameters, rbf, 
                 scale, is_approximating_on_tangent):
        super().__init__(manifold, original_function, grid_parameters, rbf)
        self._is_approximating_on_tangent = is_approximating_on_tangent
        rbf_radius = scale
        
        self._grid = SamplingPointsCollection(rbf_radius, 
            original_function,
            grid_parameters,
            phi_generator=self._calculate_phi)

        self._kernel = generate_kernel(self._rbf, rbf_radius)

    def _calculate_phi(self, x_0, y_0):
        point = np.array([x_0, y_0])

        @cached(cache=generate_cache(maxsize=10))
        def phi(x, y):
            vector = np.array([x, y])
            return self._kernel(vector, point)

        return phi
    
    @cached(cache=generate_cache(maxsize=10))
    def approximation(self, x, y):
        """ Average sampled points around (x, y), using phis as weights """
        values_to_average = list()
        weights = list()

        for point in self._grid.points_in_radius(x, y):
            values_to_average.append(point.evaluation)
            weights.append(point.phi(x, y))

        normalizer = sum(weights)

        if normalizer == 0:
            normalizer = 0.00001

        weights = [w_i / normalizer for w_i in weights]
        if self._is_approximating_on_tangent:
            return sum(w_i * x_i for w_i, x_i in zip(weights, values_to_average))

        return manifold.average(values_to_average, weights)
