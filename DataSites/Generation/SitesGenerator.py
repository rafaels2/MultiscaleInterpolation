"""
Not implemented yet
"""
from abc import abstractmethod


class SitesGeneration(object):
    # TODO: do this
    def __init__(self, grid_parameters):
        self._data_points = self._generate_points(grid_parameters)
        pass

    @abstractmethod
    def _generate_points(self, grid_parameters):
        pass

    @property
    def data_points(self):
        return self._data_points
