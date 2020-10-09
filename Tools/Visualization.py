"""
inspired by https://stackoverflow.com/questions/41955492/
how-to-plot-efficiently-a-large-number-of-3d-ellipsoids-with-matplotlib-axes3d
"""
from abc import abstractmethod

import numpy as np
from numpy import linalg as la

from matplotlib import cm
import matplotlib.pyplot as plt

from pytransform3d.rotations import plot_basis

if __name__ == "__main__":
    from Manifolds.SymmetricPositiveDefinite import SymmetricPositiveDefinite
    from Manifolds.RigidRotations import RigidRotations


class Visualizer(object):
    def __init__(self, matrices, centers):
        self.fig = plt.figure(figsize=(8, 8))
        # TODO: self.ax = self.fig.add_subplot(projection='3d')
        # TODO: change this when visualizing 3D.
        # TODO: self.ax.view_init(azim=0, elev=90)
        self._matrices = matrices
        self._centers = centers

    @abstractmethod
    def _process_matrix(self, index):
        pass

    def save(self, filename, title):
        for index in np.ndindex(self._matrices.shape):
            self._process_matrix(index)
        self._post_process()
        # Hide axes ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        # TODO: clean
        """ 
        self.ax.set_zticks([])
        """
        # Add a color bar which maps values to colors.
        # plt.colorbar()
        plt.title(title)
        plt.savefig(filename)
        plt.close(self.fig)

    def _post_process(self):
        pass

    def show(self):
        for index in np.ndindex(self._matrices.shape):
            self._process_matrix(index)
        self._post_process()
        # Hide axes ticks
        # self.ax.set_xticks([])
        # self.ax.set_yticks([])
        # TODO: self.ax.set_zticks([])

        plt.show()


class EllipsoidVisualizer(Visualizer):
    def __init__(self, matrices, centers):
        super().__init__(matrices, centers)

        self._svd_matrices()
        self._calculate_normalizer()

    def _calculate_normalizer(self):
        max_radius = 0
        for index in np.ndindex(self._singular_values.shape):
            radii = self._singular_values[index]
            if max(radii) > max_radius:
                max_radius = max(radii)

        print("Max Radius is ", max_radius)

        # TODO: normalize according to min fill distance
        self._normalizer = max_radius * 2

    def _svd_matrices(self):
        singular_values = np.zeros_like(self._matrices, dtype=object)
        rotations = np.zeros_like(self._matrices)
        for index in np.ndindex(self._matrices.shape):
            _, singular_values[index], rotations[index] = la.svd(self._matrices[index])

        self._singular_values = singular_values
        self._rotations = rotations

    def _process_matrix(self, index):
        center = self._centers[index]
        radii = self._singular_values[index]
        rotation = self._rotations[index]

        # calculate cartesian coordinates for the ellipsoid surface
        u = np.linspace(0.0, 2.0 * np.pi, 60)
        v = np.linspace(0.0, np.pi, 60)
        x = radii[0] * np.outer(np.cos(u), np.sin(v)) / self._normalizer
        y = radii[1] * np.outer(np.sin(u), np.sin(v)) / self._normalizer
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) / self._normalizer

        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

        self.ax.plot_surface(x, y, z, rstride=3, cstride=3, linewidth=0.1, alpha=1, shade=True,
                             cmap=cm.coolwarm)


def _calculate_direction(radii, rotation):
    axis = np.zeros((3, ))
    axis[radii.argmax()] = 1
    return np.matmul(rotation, axis)


def _calculate_fa(radii):
    mean_diffusivity = sum(radii) / 3
    num = la.norm(radii - mean_diffusivity * np.ones((3,)), 2)
    den = la.norm(radii, 2)
    return np.sqrt(3/2) * num / den


class BrainVisualizer(EllipsoidVisualizer):
    def __init__(self, matrices, centers, max_size, paint_radius):
        self._max_size = max_size
        centers = centers.astype(int)
        self._color_map = np.zeros((max_size, max_size, 3))
        self._counter = np.zeros((max_size, max_size), dtype=int)
        self._paint_radius = int(paint_radius)
        super(BrainVisualizer, self).__init__(matrices, centers)

    def _paint(self, center, color):
        center_index = tuple(center)
        self._color_map[center_index] = color
        self._counter[center_index] = -1
        for index in np.ndindex((self._paint_radius, self._paint_radius)):
            current_index = tuple(center + index)
            if any(i < 0 or i >= self._max_size for i in current_index):
                continue
            if self._counter[current_index] < 0:
                continue

            self._counter[current_index] += 1
            den = float(self._counter[current_index])

            self._color_map[current_index] = (self._color_map[current_index] * (self._counter[current_index] - 1) +
                                                  color) / den

    def _process_matrix(self, index):
        center = self._centers[index]
        radii = self._singular_values[index]
        rotation = self._rotations[index]

        color = (_calculate_fa(radii) * _calculate_direction(radii, rotation) + 1) / 2
        self._paint(center, color)

    def _post_process(self):
        plt.imshow(self._color_map)


class RotationVisualizer(Visualizer):
    """docstring for RotationVisualizer"""

    def __init__(self, matrices, centers):
        super().__init__(matrices, centers)
        min_lim, max_lim = self._get_limits()
        plt.setp(self.ax, xlim=(min_lim, max_lim), ylim=(min_lim, max_lim), zlim=(min_lim, max_lim))

    def _get_limits(self):
        if self._centers.dtype.kind == 'f':
            centers = self._centers
        else:
            shape = list(self._centers.shape)
            shape.append(3)
            centers = np.zeros(shape)
            if self._centers.dtype != np.float:
                for index in np.ndindex(centers.shape):
                    centers[index] = self._centers[index[:-1]][index[-1]]

        # noinspection PyArgumentList
        return centers.min() - 1, centers.max() + 1

    def _process_matrix(self, index):
        center = self._centers[index]
        matrix = self._matrices[index]
        plot_basis(ax=self.ax, R=matrix, p=center, s=0.4)


def ellipsoids_main():
    print("start")
    spd = SymmetricPositiveDefinite()
    matrices = np.zeros((3, 3), dtype=object)
    centers = np.zeros_like(matrices)
    for index in np.ndindex(matrices.shape):
        matrices[index] = spd.gen_point()
        centers[index] = np.array([index[0], index[1], 0])
    print("start to visualize")
    EllipsoidVisualizer(matrices, centers).save("vis.png", "ellipsoids")


def rotations_main():
    print("start")
    so_3 = RigidRotations()
    matrices = np.zeros((3, 3), dtype=object)
    centers = np.zeros_like(matrices)
    for index in np.ndindex(matrices.shape):
        centers[index] = np.array([index[0], index[1], 0])
        matrices[index] = so_3.gen_point()

    print("start to visualize")
    RotationVisualizer(matrices, centers).save("rotations.png", "rotations")


if __name__ == "__main__":
    rotations_main()
