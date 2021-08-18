"""
This structure saves the data assuming it's in a grid.
The query just assumes that the block of size {rbf_radius} has all closest points.
"""
import numpy as np

from DataSites.PolynomialReproduction import PolynomialReproduction
from . import add_sampling_class
from DataSites.Storage.Storage import DataSitesStorage, Point


@add_sampling_class("grid")
class Grid(DataSitesStorage):
    # TODO: This should be a data generation method
    def __init__(
        self,
        sites,
        rbf_radius,
        function_to_evaluate,
        fill_distance,
        phi_generator=None,
    ):
        self._x, self._y = sites
        self._x_min = np.min(self._x)
        self._y_min = np.min(self._y)
        self._evaluation = self._evaluate_on_grid(function_to_evaluate)
        self._phi = None
        self._fill_distance = fill_distance

        if phi_generator is not None:
            self._phi = self._evaluate_on_grid(phi_generator)

        self._radius_in_index = int(np.ceil(rbf_radius / self._fill_distance))

        self._lambdas_generator = PolynomialReproduction(self, "grid_cache.pkl")
        self._lambdas = self._evaluate_on_grid(self._lambdas_generator.weight_for_grid)

    def _evaluate_on_grid(self, func):
        evaluation = np.zeros_like(self._x, dtype=object)

        for index in np.ndindex(self._x.shape):
            if index[1] == 0:
                print(index[0] / self._x.shape[0])
            evaluation[index] = func(self._x[index], self._y[index])

        return evaluation

    def points_in_radius(self, x, y):
        # Warning! There might be a bug, and I should want to replace x, and y.
        x_0 = int((x - self._x_min) / self._fill_distance)
        y_0 = int((y - self._y_min) / self._fill_distance)
        index_0 = np.array([y_0, x_0])
        radius_array = np.array([self._radius_in_index + 1, self._radius_in_index + 1])

        for index in np.ndindex(
            2 * self._radius_in_index + 2, 2 * self._radius_in_index + 2
        ):
            current_index = tuple(index_0 - radius_array + np.array(index))
            if all(
                [
                    current_index[0] >= 0,
                    current_index[1] >= 0,
                    current_index[0] < self._x.shape[0],
                    current_index[1] < self._y.shape[1],
                ]
            ):
                # This is important to avoid singular matrices.
                if self._phi[current_index](x, y) != 0:
                    yield Point(
                        self._evaluation[current_index],
                        self._phi[current_index],
                        self._x[current_index],
                        self._y[current_index],
                        self._lambdas[current_index],
                    )

    @property
    def evaluation(self):
        return self._evaluation

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def close(self):
        self._lambdas_generator.update()
