"""
This method uses the kd-tree algorithm.
"""
import numpy as np
from numpy import linalg as la
from pykdtree.kdtree import KDTree

from Config.Config import config
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
        self._evaluation = self._evaluate_on_grid(function_to_evaluate)
        if config.IS_APPROXIMATING_ON_TANGENT:
            error_function = lambda x, y: la.norm(function_to_evaluate(x, y))
        else:
            error_function = lambda x, y: config.MANIFOLD.distance(
                function_to_evaluate(x, y), config.MANIFOLD.zero_func(x, y)
            )
        self._error_evaluation = self._evaluate_on_grid(error_function)
        self._max_error = np.max(np.abs(self._error_evaluation.ravel()))
        self._seq, self._evaluation = self._denoise()
        self._tree = KDTree(self._seq)
        self._phi = None

        # TODO: test for the case of quadratic reproduction
        self._lambdas_generator = PolynomialReproduction(self, "grid_cache.pkl")
        self._lambdas = self._evaluate_on_grid(self._lambdas_generator.weight_for_grid)

        if phi_generator is not None:
            self._phi = self._evaluate_on_grid(phi_generator)
            self._phi_generator = phi_generator

    def _denoise(self):
        if not config.DENOISE or config.scale_index <= 1:
            return self._seq, self._evaluation

        seq = list()
        evaluation = list()
        for index in np.ndindex(self._evaluation.shape):
            if (
                self._error_evaluation[index]
                <= config.DENOISE_THRESHOLD * self._max_error
            ):
                seq.append(self._seq[index])
                evaluation.append(self._evaluation[index])
        seq = np.array(seq)
        evaluation = np.array(evaluation)
        return seq, evaluation

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

    def _evaluate_on_grid(self, function_to_evaluate):
        evaluation = np.zeros(self._seq.shape[0], dtype=object)

        for index in range(evaluation.shape[0]):
            evaluation[index] = function_to_evaluate(
                self._seq[index, 0], self._seq[index, 1]
            )

        return evaluation

    @property
    def x(self):
        return self._seq[:, 0]

    @property
    def y(self):
        return self._seq[:, 1]
