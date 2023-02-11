"""
This is the basic approximation method.
To implement it one should inherit, and implement the approximation.
"""
from abc import abstractmethod
import numpy as np
from cachetools import cached
from matplotlib import pyplot as plt

from Tools.Utils import generate_cache


class ApproximationMethod(object):
    def __init__(self, manifold, original_function, grid_parameters, rbf):
        self._original_function = original_function
        self._grid_parameters = grid_parameters
        self._rbf = rbf
        self._manifold = manifold

    @abstractmethod
    def approximation(self, x, y):
        pass

    @property
    @abstractmethod
    def data_sites(self):
        pass

    def _calculate_phi(self, x_0, y_0):
        point = np.array([x_0, y_0])

        @cached(cache=generate_cache(maxsize=100))
        def phi(x, y):
            vector = np.array([x, y])
            return self._kernel(vector, point)

        return phi

    def plot_sites(self):
        plt.figure()
        fig = plt.scatter(self.data_sites.x, self.data_sites.y, c="#000000", marker="+")
        xaxis = fig.axes.get_xaxis()
        xaxis.set_visible(False)
        # ans1 = fig.axes.get_xlim()
        # fig.axes.set_xlim(self._grid_parameters.x_min, self._grid_parameters.x_max)
        # ans2 = fig.axes.get_xlim()
        # ans3 = fig.axes.get_ylim()
        yaxis = fig.axes.get_yaxis()
        # ans4 = fig.axes.get_xlim()
        yaxis.set_visible(False)
        # fig.axes.set_ylim(self._grid_parameters.y_min, self._grid_parameters.y_max)
        # aspect = fig.axes.get_aspect()
        fig.axes.set_aspect(1)
        plt.savefig("sites_scatter.png", bbox_inches="tight")
