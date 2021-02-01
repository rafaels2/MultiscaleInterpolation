from abc import abstractmethod
import numpy as np
from cachetools import cached

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

	def _calculate_phi(self, x_0, y_0):
		point = np.array([x_0, y_0])

		@cached(cache=generate_cache(maxsize=100))
		def phi(x, y):
			vector = np.array([x, y])
			return self._kernel(vector, point)

		return phi
