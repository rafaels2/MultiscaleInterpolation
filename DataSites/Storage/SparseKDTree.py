import numpy as np
from pykdtree.kdtree import KDTree

from DataSites.Storage import add_sampling_class
from DataSites.Storage.KDTree import KDTreeSampler
from Config.Config import config
from DataSites.Storage.Storage import Point


@add_sampling_class("sparse-kd-tree")
class SparseKDTree(KDTreeSampler):
    def __init__(self, sites, rbf_radius, function_to_evaluate, *_, phi_generator=None):
        super(SparseKDTree, self).__init__(sites, rbf_radius, function_to_evaluate, _, phi_generator=phi_generator)
        self._full_sequence = config.SEQUENCE
        self._full_kd_tree = KDTree(self._full_sequence)
        self._function_to_evaluate = function_to_evaluate

    def points_in_radius(self, x, y):
        counter = 0
        for p in super(SparseKDTree, self).points_in_radius(x, y):
            yield p

        if counter < 2:
            _, idx = self._full_kd_tree.query(np.array([[x, y]]), k=3)
            for index in idx[0]:
                x = self._full_sequence[index, 0]
                y = self._full_sequence[index, 1]
                phi = self._phi_generator(x, y)
                func = self._function_to_evaluate(x, y)
                yield Point(func, phi, x, y, 0)
