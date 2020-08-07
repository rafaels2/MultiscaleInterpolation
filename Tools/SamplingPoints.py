from abc import abstractmethod
from collections import namedtuple
import numpy as np
from tqdm import tqdm


GridParameters = namedtuple('GridParameters', ['x_min', 'x_max', 'y_min', 'y_max', 'mesh_norm'])
Point = namedtuple('Point', ['evaluation', 'phi', 'x', 'y'])

SAMPLING_POINTS_CLASSES = dict()


def generate_grid(grid_size, resolution, scale=1, should_ravel=True):
    print("creating a grid", 2 * resolution / scale)
    y = np.linspace(-grid_size, grid_size, int(2 * resolution / scale))
    x = np.linspace(-grid_size, grid_size, int(2 * resolution / scale))
    x_matrix, y_matrix = np.meshgrid(x, y)
    if should_ravel:
        return x_matrix.ravel(), y_matrix.ravel()
    else:
         return x_matrix, y_matrix


def symmetric_grid_params(grid_size, mesh_norm):
    return GridParameters(-grid_size, grid_size, -grid_size, grid_size, mesh_norm)


def add_sampling_class(name):
    def _register_decorator(cls):
        SAMPLING_POINTS_CLASSES[name] = cls
        return cls
    return _register_decorator


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
    def __init__(self, rbf_radius, function_to_evaluate, grid_parameters, phi_generator=None):
        self._x_min = grid_parameters.x_min
        self._x_max = grid_parameters.x_max
        self._y_min = grid_parameters.y_min
        self._y_max = grid_parameters.y_max
        self._x_len = (self._x_max - self._x_min)
        self._y_len = (self._y_max - self._y_min)
        self._mesh_norm = grid_parameters.mesh_norm
        print("Mesh norm: ", self._mesh_norm)
        self._x, self._y = self._generate_grid()
        self._evaluation = self._evaluate_on_grid(function_to_evaluate)
        self._phi = None

        if phi_generator is not None:
            self._phi = self._evaluate_on_grid(phi_generator)
        
        self._radius_in_index = int(np.ceil(rbf_radius / self._mesh_norm))

    def _generate_grid(self):
        x = np.linspace(self._x_min, self._y_max, int(self._x_len / self._mesh_norm))
        y = np.linspace(self._y_min, self._y_max, int(self._y_len / self._mesh_norm))
        return np.meshgrid(x, y)

    def _evaluate_on_grid(self, func):
        evaluation = np.zeros_like(self._x, dtype=object)
        
        for index in tqdm(np.ndindex(self._x.shape)):
            evaluation[index] = func(self._x[index], self._y[index])

        return evaluation

    def points_in_radius(self, x, y):
        # Warning! There might be a bug, and I should want to replace x, and y.
        x_0 = int((x - self._x_min) / self._mesh_norm)
        y_0 = int((y - self._y_min) / self._mesh_norm)
        index_0 = np.array([y_0, x_0])
        radius_array = np.array([self._radius_in_index + 1, self._radius_in_index + 1])

        for index in np.ndindex((2 * self._radius_in_index + 2, 2 * self._radius_in_index + 2)):
            current_index = tuple(index_0 - radius_array + np.array(index))
            if all([current_index[0] >= 0, current_index[1] >= 0, 
                    current_index[0] < self._x.shape[0], current_index[1] < self._y.shape[1]]):
                yield Point(self._evaluation[current_index], self._phi[current_index], self._x[current_index], self._y[current_index])

    @property
    def evaluation(self):
        return self._evaluation


class SamplingPointsCollection(object):
    def __init__(self, rbf_radius, function_to_evaluate, grids_parameters, **kwargs):
        self._grids = [SAMPLING_POINTS_CLASSES[name](rbf_radius, 
                                                     function_to_evaluate, parameters, **kwargs) 
                       for name, parameters in grids_parameters]

    def points_in_radius(self, x, y):
        for grid in self._grids:
            yield from grid.points_in_radius(x, y)


def main():
    def func(x, y):
        return x
    def phi(x, y):
        def f(a, b):
            return (a-x)**2 + (b-y)**2
        return f
    grid_parameters = GridParameters(-1, 1, -1, 1, 0.2)
    collection_params = [('Grid', grid_parameters)]
    smpl = SamplingPointsCollection(0.5, func, collection_params, phi_generator=phi)
    return smpl


if __name__ == '__main__':
    smpl = main()
