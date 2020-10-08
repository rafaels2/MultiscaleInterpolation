from abc import abstractmethod


class ApproximationMethod(object):
	def __init__(self, manifold, original_function, grid_parameters, rbf):
		self._original_function = original_function
		self._grid_parameters = grid_parameters
		self._rbf = rbf
		self._manifold = manifold

	@abstractmethod
	def approximation(self, x, y):
		# TODO: change (x, y) to p.
		pass
