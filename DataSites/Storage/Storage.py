from abc import abstractmethod
from collections import namedtuple

import numpy as np
from cachetools import cached

from DataSites.Generation.Combination import SamplingPointsCollection
from DataSites.GridUtils import GridParameters
from DataSites.PolynomialReproduction import PolynomialReproduction
from Tools.Utils import generate_cache, generate_kernel
from RBF import wendland_3_1

Point = namedtuple("Point", ["evaluation", "phi", "x", "y", "lambdas"])


class DataSitesStorage(object):
    # TODO: do this

    def __init__(self, sites, rbf_radius, function_to_evaluate, *args, **kwargs):
        # TODO: remove rbf_radius
        # TODO: change to "functions to evaluate" (and now no need to phi_generator).
        # TODO: maybe function_to_evaluate can be in a setter.
        # TODO: add to grid_parameters the method of creation
        self._rbf_radius = rbf_radius
        self._function_to_evaluate = function_to_evaluate

    @abstractmethod
    def points_in_radius(self, x, y):
        # TODO: change to (point, radius)
        pass


# TODO: A multiscale class that aggregates points. (add_points method)


def unittest():
    def _calculate_phi(x_0, y_0):
        point = np.array([x_0, y_0])
        rbf_radius = 0.5

        @cached(cache=generate_cache(maxsize=100))
        def phi(x, y):
            vector = np.array([x, y])
            kernel = generate_kernel(wendland_3_1, rbf_radius)
            return kernel(vector, point)

        return phi

    def func(x, y):
        return x

    grid_parameters = GridParameters(-1, 1, -1, 1, 0.2)
    collection_params = [("Grid", grid_parameters)]
    grid = SamplingPointsCollection(
        0.5, func, collection_params, phi_generator=_calculate_phi
    )
    lambdas = PolynomialReproduction(grid, "test.pkl")

    lambdas_0 = lambdas.calculate(0, 0)
    lambdas_0_5 = lambdas.calculate(0, 0.5)
    lambdas.update()
    return grid, lambdas_0, lambdas_0_5, lambdas


def main():
    unittest()


if __name__ == "__main__":
    main()
