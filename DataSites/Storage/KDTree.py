"""
This method uses the kd-tree algorithm.
"""
import numpy as np
from pykdtree.kdtree import KDTree

from DataSites.PolynomialReproduction import PolynomialReproduction
from DataSites.Storage import add_sampling_class
from DataSites.Storage.Storage import DataSitesStorage, Point


@add_sampling_class("kd-tree")
class KDTreeSampler(DataSitesStorage):
    # TODO: add the other tree method from Wendland's book.
    def __init__(self, sites, rbf_radius, function_to_evaluate, *_, phi_generator=None):

        # Grid compatability
        if type(sites) is tuple:
            points_in_matrices = [axis.ravel() for axis in sites]
            sites = np.transpose(np.array(points_in_matrices))

        self._rbf_radius = rbf_radius
        self._seq = sites
        self._tree = KDTree(self._seq)
        self._evaluation = self._evaluate_on_grid(function_to_evaluate)
        self._phi = None

        # TODO: test for the case of quadratic reproduction
        self._lambdas_generator = PolynomialReproduction(self, "grid_cache.pkl")
        self._lambdas = self._evaluate_on_grid(self._lambdas_generator.weight_for_grid)

        if phi_generator is not None:
            self._phi = self._evaluate_on_grid(phi_generator)

    def points_in_radius(self, x, y):
        p = np.array([[x, y]])
        _, ids = self._tree.query(p, k=30, distance_upper_bound=self._rbf_radius)
        indices = ids[0]
        last_index = 0

        if indices[-1] < self._seq.shape[0]:
            print("reached K")

        for index in indices:
            if index == self._seq.shape[0]:
                break

            yield Point(
                self._evaluation[index],
                self._phi[index],
                self._seq[index, 0],
                self._seq[index, 1],
                self._lambdas[index],
            )
            last_index += 1
        print(last_index)

    def _evaluate_on_grid(self, function_to_evaluate):
        evaluation = np.zeros(self._seq.shape[0], dtype=object)

        for index in range(evaluation.shape[0]):
            evaluation[index] = function_to_evaluate(
                self._seq[index, 0], self._seq[index, 1]
            )

        return evaluation
