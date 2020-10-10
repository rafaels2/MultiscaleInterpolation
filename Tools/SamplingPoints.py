from abc import abstractmethod
from collections import namedtuple

import numpy as np
from numpy import linalg as la
from tqdm import tqdm

GridParameters = namedtuple('GridParameters', ['x_min', 'x_max', 'y_min', 'y_max', 'mesh_norm'])
Point = namedtuple('Point', ['evaluation', 'phi', 'x', 'y'])
Evaluation = namedtuple('Evaluation', 'data, centers')

SAMPLING_POINTS_CLASSES = dict()
SCORE_THRESHOLD = 1


def generate_grid(grid_size, resolution, scale=1, should_ravel=True):
    """
    This is a utility function that generates matching x, y points array for a grid.
    :param grid_size: distance of border from 0
    :param resolution: TODO describe
    :param scale:
    :param should_ravel:
    :return:
    """
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
    def __init__(self, rbf_radius, function_to_evaluate, grid_parameters, phi_generator=None, *args, **kwargs):
        super().__init__(rbf_radius, function_to_evaluate, *args, **kwargs)
        self._x_min = grid_parameters.x_min
        self._x_max = grid_parameters.x_max
        self._y_min = grid_parameters.y_min
        self._y_max = grid_parameters.y_max
        self._mesh_norm = grid_parameters.mesh_norm
        self._x, self._y = self._generate_grid()
        self._evaluation = self.evaluate_on_grid(function_to_evaluate)
        self._phi = None

        if phi_generator is not None:
            self._phi = self.evaluate_on_grid(phi_generator)

        self._radius_in_index = int(np.ceil(rbf_radius / self._mesh_norm))

    def _generate_grid(self):
        try:
            x = np.linspace(self._x_min, self._x_max, int((self._x_max - self._x_min) / self._mesh_norm) + 1)
            y = np.linspace(self._y_min, self._y_max, int((self._y_max - self._y_min) / self._mesh_norm) + 1)
        except:
            import ipdb; ipdb.set_trace()
        return np.meshgrid(x, y)

    def evaluate_on_grid(self, func):
        evaluation = np.zeros_like(self._x, dtype=object)

        print(f"X shape: {self._x.shape}")
        for index in tqdm(np.ndindex(self._x.shape)):
            if index[- 1] == 0:
                print(index[0] / self._x.shape[0])
            evaluation[index] = func(self._x[index], self._y[index])

        return evaluation

    def _is_in_radius(self, x, y, current_index):
        return la.norm(np.array([x - self._x[current_index], y - self._y[current_index]]), 2) < self._rbf_radius

    def points_in_radius(self, x, y):
        # Warning! There might be a bug, and I should want to replace x, and y.
        x_0 = int((x - self._x_min) / self._mesh_norm)
        y_0 = int((y - self._y_min) / self._mesh_norm)
        index_0 = np.array([y_0, x_0])
        radius_array = np.array([self._radius_in_index + 1, self._radius_in_index + 1])

        for index in np.ndindex((2 * self._radius_in_index + 2, 2 * self._radius_in_index + 2)):
            current_index = tuple(index_0 - radius_array + np.array(index))
            if all([current_index[0] >= 0, current_index[1] >= 0,
                    current_index[0] < self._x.shape[0], current_index[1] < self._y.shape[1],
                    self._is_in_radius(x, y, current_index)]):
                yield Point(self._evaluation[current_index], self._phi[current_index], self._x[current_index],
                            self._y[current_index])

    @property
    def evaluation(self):
        return Evaluation(self._evaluation, np.array(list(zip(self._x, self._y))))


class SubDomain(Grid):
    """ This can be used for final evaluation """
    def __init__(self, confidence_score, *args, **kwargs):
        self._confidence_score = confidence_score
        super().__init__(*args, **kwargs)

    def _generate_grid(self):
        _x, _y = super()._generate_grid()
        x = list(_x.astype(int).ravel())
        y = list(_y.astype(int).ravel())
        indices_to_pop = list()

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            if self._confidence_score(x_i, y_i) < SCORE_THRESHOLD:
                indices_to_pop.append(i)

        # I don't sort here indices_to_pop because I know its order, but it's important.
        while len(indices_to_pop) > 0:
            i = indices_to_pop.pop()
            x.pop(i)
            y.pop(i)

        return np.array(x), np.array(y)

    def points_in_radius(self, x, y):
        for index in np.ndindex(self._x.shape):
            if self._is_in_radius(x, y, index):
                yield Point(self._evaluation[index], self._phi[index], self._x[index],
                            self._y[index])


@add_sampling_class('DynamicGrid')
class DynamicGrid(Grid):
    def __init__(self, confidence_score, *args, **kwargs):
        self._phi_generator = kwargs.get('phi_generator', None)
        kwargs['phi_generator'] = None
        super().__init__(*args, **kwargs)

        self._confidence_score = confidence_score
        self._sub_domains = self._init_sub_domains()

    def _generate_grid(self):
        _x, _y = super(DynamicGrid, self)._generate_grid()
        return _x.astype(int), _y.astype(int)

    def _init_sub_domains(self, ):
        sub_x = self._x[::self._radius_in_index, ::self._radius_in_index]
        sub_y = self._y[::self._radius_in_index, ::self._radius_in_index]
        sub_domains = np.zeros(tuple(np.array(sub_x.shape) - np.array([1, 1])), dtype=object)
        for index in np.ndindex(tuple(np.array(sub_x.shape) - np.array([1, 1]))):
            # TODO: Debug this - [0, 1] can be [1, 0].
            grid_parameters = GridParameters(sub_x[index],
                                             sub_x[tuple(np.array(index) + np.array([0, 1]))],
                                             sub_y[index],
                                             sub_y[tuple(np.array(index) + np.array([1, 0]))],
                                             self._mesh_norm)
            sub_domains[index] = SubDomain(self._confidence_score, self._rbf_radius, self._function_to_evaluate,
                                           grid_parameters, phi_generator=self._phi_generator)

        return sub_domains

    def evaluate_on_grid(self, func):
        # TODO: implement dynamic grid evaluation
        # Split to boxes and hold lists of areas + boundary
        return None

    def points_in_radius(self, x, y):
        # Warning! There might be a bug, and I should want to replace x, and y.
        x_0 = int((x - self._x_min) / self._rbf_radius)
        y_0 = int((y - self._y_min) / self._rbf_radius)
        index_0 = np.array([y_0, x_0])
        # 3X3 around a radius sized square is enough.
        radius_array = np.array([3, 3])

        for index in np.ndindex(tuple(radius_array)):
            current_index = tuple(index_0 - np.array([1, 1]) + np.array(index))
            # TODO: debug this
            if all([current_index[0] >= 0, current_index[1] >= 0,
                    current_index[0] < self._sub_domains.shape[0],
                    current_index[1] < self._sub_domains.shape[1]]):
                yield from self._sub_domains[current_index].points_in_radius(x, y)

    @property
    def evaluation(self):
        raise NotImplemented('This grid is used for approximation step')


class SamplingPointsCollection(object):
    def __init__(self, rbf_radius, function_to_evaluate, confidence, grids_parameters, **kwargs):
        # TODO: use confidence inside grid_parameters
        self._grids = [SAMPLING_POINTS_CLASSES[name](confidence, rbf_radius,
                                                     function_to_evaluate, parameters, **kwargs)
                       for name, parameters in grids_parameters]

    def points_in_radius(self, x, y):
        for grid in self._grids:
            yield from grid.points_in_radius(x, y)


def main():
    def func(x, y):
        return x + y

    def phi(x, y):
        def f(a, b):
            return (a - x) ** 2 + (b - y) ** 2

        return f

    grid_parameters = GridParameters(-1, 1, -1, 1, 0.2)
    collection_params = [('Grid', grid_parameters)]
    _smpl = SamplingPointsCollection(0.5, func, collection_params, phi_generator=phi)
    return _smpl


if __name__ == '__main__':
    smpl = main()
