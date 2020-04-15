import numpy as np
from cachetools import cached
from numpy import linalg as la
from collections import namedtuple
from Tools.Utils import generate_grid, generate_kernel, evaluate_on_grid, generate_cache

from .ApproximationMethod import ApproximationMethod

Grid = namedtuple('Grid', ['x', 'y'])


class Quasi(ApproximationMethod):
    def __init__(self, manifold, original_function, grid_parameters, rbf, is_approximating_on_tangent):
        super().__init__(manifold, original_function, grid_parameters, rbf)
        self._is_approximating_on_tangent = is_approximating_on_tangent

        self._grid = Grid(*generate_grid(*self._grid_parameters, should_ravel=False))
        self._kernel = generate_kernel(self._rbf, self._grid_parameters.scale)
        self._mesh_norm = (2 * self._grid_parameters.size / self._grid.x.shape[0])
        self._radius_in_index = int(np.ceil(self._grid_parameters.scale / self._mesh_norm))

        self._init_phis()
        self._values_on_grid = evaluate_on_grid(self._original_function, points=self._grid)

    def _calculate_phi(self, i, j):
        point = np.array([self._grid.x[i, j], self._grid.y[i, j]])

        @cached(cache=generate_cache(maxsize=10))
        def phi(x, y):
            vector = np.array([x, y])
            return self._kernel(vector, point)

        return phi
    
    def _init_phis(self):
        """ Initialize RBFs that centered around sampled points"""
        self._phis = np.array([[self._calculate_phi(i, j) for j in range(self._grid.x.shape[1])]
            for i in range(self._grid.x.shape[0])])

    def _index_from_offset(self, x_0, y_0, offset):
        """ Translates offset and p_0 to a index in the grid"""
        return Grid(int(y_0 - self._radius_in_index - 1 + offset[1]),
            int(x_0 - self._radius_in_index - 1 + offset[0]))

    def _supported_indices(self, x_0, y_0):
        """ Returns a list of the supported points (according to scale) around p_0 """
        indices = (self._index_from_offset(x_0, y_0, offset) for offset in
            np.ndindex((2 * self._radius_in_index + 2, 2 * self._radius_in_index + 2)))

        return [(y, x) for y, x in indices 
            if all([x >= 0, y >= 0, x < self._phis.shape[0], y < self._phis.shape[1]])]

    @cached(cache=generate_cache(maxsize=10))
    def approximation(self, x, y):
        """ Average sampled points around (x, y), using phis as weights """
        min_value = -self._grid_parameters.size
        x_0 = (x - min_value) / self._mesh_norm
        y_0 = (y - min_value) / self._mesh_norm

        supported_indices = self._supported_indices(x_0, y_0)
        values_to_average = [self._values_on_grid[index] for index in supported_indices] 
        weights = [self._phis[index](x, y) for index in supported_indices]
        normalizer = sum(weights)

        if normalizer == 0:
            normalizer = 0.00001

        weights = [w_i / normalizer for w_i in weights]
        if self._is_approximating_on_tangent:
            return sum(w_i * x_i for w_i, x_i in zip(weights, values_to_average))

        return manifold.average(values_to_average, weights)
