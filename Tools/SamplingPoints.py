from abc import abstractmethod
from collections import namedtuple
import numpy as np

from Tools.Utils import generate_grid


GridParameters = namedtuple('GridParameters', ['x_min', 'x_max', 'y_min', 'y_max', 'mesh_norm'])
Point = namedtuple('Point', ['evaluation', 'phi'])

SAMPLING_POINTS_CLASSES = dict()


def add_sampling_class(name):
	def _register_decorator(cls):
		SAMPLING_POINTS_CLASSES[name] = cls
		return cls


class SamplingPoints(object):
	"""docstring for SamplingPoints"""
	def __init__(self, rbf_radius, function_to_evaluate, *args, **kwargs):
		self._rbf_radius = rbf_radius
		self._function_to_evaluate = function_to_evaluate

	@abstractmethod
	def points_in_radius(self, x, y):
		pass

@add_sampling_class('Grid')
class Grid(SamplingPoints):
	def __init__(self, rbf_radius, function_to_evaluate, grids_parameters, phi_generator=None):
		self._x_min = grids_parameters.x_min
		self._x_max = grids_parameters.x_max
		self._y_min = grids_parameters.y_min
		self._y_max = grids_parameters.y_max
		self._max = grid_parameters.max_value
		self._x_len = (self._x_max - self._x_min)
		self._y_len = (self._y_max - self._y_min)
		self._mesh_norm = grid_parameters.mesh_norm
		self._x, self._y = self._generate_grid()
		self._evaluation = self._evaluate_on_grid(function_to_evaluate)
		self._phi = None

		if phi_generator is not None:
			self._phi = self._evaluate_on_grid(phi_generator)
		
		self._radius_in_index = int(np.ceil(rbf_radius / self._mesh_norm))

	def _generate_grid(self):
		x = np.linspace(self._min, self._max, int(self._x_len / self._mesh_norm))
		y = np.linspace(self._min, self._max, int(self._y_len / self._mesh_norm))
    	return np.meshgrid(x, y)

	def _evaluate_on_grid(self, func):
		evaluation = np.zeros_like(self._x, dtype=object)
		
		for index in np.ndindex(self._x.shape):
			evaluation[index] = func(self._x[index], self._y[index])

		return evaluation

	def points_in_radius(self, x, y):
		# Warning! There might be a bug, and I should want to replace x, and y.
		x_0 = (x - self._x_min) / self._mesh_norm
		y_0 = (y - self._y_min) / self._mesh_norm
		index_0 = np.array([x_0, y_0])
		radiud_array = np.array([self._radius_in_index + 1, self._radius_in_index + 1])

		for index in np.ndindex((2 * self._radius_in_index + 2, 2 * self._radius_in_index + 2)):
			current_index = index_0 - radiud_array + index
			if all([current_index[0] >= 0, current_index[1] >= 0, 
					current_index[0] < self._x.shape, current_index[1] < self._y.shape]):
				yield Point(self._evaluation[index], self._phi[index])


class SamplingPointsCollection(object):
	def __init__(self, rbf_radius, function_to_evaluate, grids_parameters, **kwargs):
		self._grids = [SAMPLING_POINTS_CLASSES[name](rbf_radius, 
													 function_to_evaluate, parameters, **kwargs) 
					   for name, parameters in grids_parameters]

	def points_in_radius(self, x_0, y_0):
		for grid in self._grids:
			yield from grid.points_in_radius()
