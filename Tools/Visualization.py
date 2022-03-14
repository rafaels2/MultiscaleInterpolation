"""
inspired by https://stackoverflow.com/questions/41955492/
how-to-plot-efficiently-a-large-number-of-3d-ellipsoids-with-matplotlib-axes3d
"""
from abc import abstractmethod

import numpy as np
from numpy import linalg as la

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

from pytransform3d.rotations import plot_basis

if __name__ == "__main__":
    from Manifolds.SymmetricPositiveDefinite import SymmetricPositiveDefinite
    from Manifolds.RigidRotations import RigidRotations


VISUALIZATION_CONST = 10
MAX_AXIS_LEN = 15


class Visualizer(object):
    def __init__(self, matrices, centers, dims=3):
        self.fig = plt.figure(figsize=(12, 12))
        if dims == 3:
            self.ax = self.fig.add_subplot(projection="3d")
            self.ax.view_init(azim=0, elev=90)
        else:
            self.ax = self.fig.add_subplot()
        indices = self._get_indices(matrices.shape)
        self._matrices = matrices[indices]
        self._centers = centers[indices]

    @staticmethod
    def _get_indices(shape):
        new_shape = tuple(min(axis_len, MAX_AXIS_LEN) for axis_len in shape)
        indices = tuple(
            np.meshgrid(
                *(
                    [int(old_len / new_len) * index for index in range(new_len)]
                    for new_len, old_len in zip(new_shape, shape)
                )
            )
        )

        return indices

    def _process_matrix(self, index):
        pass

    def save(self, filename, title):
        for index in np.ndindex(self._matrices.shape):
            self._process_matrix(index)
        # Hide axes ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        # Add a color bar which maps values to colors.
        # plt.colorbar()
        # plt.title(title)
        plt.savefig(filename, bbox_inches="tight")
        plt.close(self.fig)

    def show(self):
        i = 1
        for index in np.ndindex(self._matrices.shape):
            i += 1
            if i % VISUALIZATION_CONST == 0:
                self._process_matrix(index)
        # Hide axes ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])

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

        self._normalizer = max_radius

    def _svd_matrices(self):
        singular_values = np.zeros_like(self._matrices, dtype=object)
        rotations = np.zeros_like(self._matrices)
        for index in np.ndindex(self._matrices.shape):
            _, singular_values[index], rotations[index] = la.svd(self._matrices[index])

        self._singular_values = singular_values
        self._rotations = rotations

    def _process_matrix(self, index):
        if not ((index[0] % 2 == 0) and (index[1] % 2 == 0)):
            return
        center = self._centers[index]
        radii = 2.4 * self._singular_values[index]
        rotation = self._rotations[index]

        # calculate cartesian coordinates for the ellipsoid surface
        u = np.linspace(0.0, 2.0 * np.pi, 60)
        v = np.linspace(0.0, np.pi, 60)
        x = radii[0] * np.outer(np.cos(u), np.sin(v)) / self._normalizer
        y = radii[1] * np.outer(np.sin(u), np.sin(v)) / self._normalizer
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) / self._normalizer

        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = (
                    np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center
                )

        self.ax.plot_surface(
            x,
            y,
            z,
            rstride=3,
            cstride=3,
            linewidth=0.1,
            alpha=1,
            shade=True,
            cmap=cm.coolwarm,
        )


class RotationVisualizer(Visualizer):
    """docstring for RotationVisualizer"""

    def __init__(self, matrices, centers):
        super().__init__(matrices, centers, dims=2)

    def save(self, filename, title):
        d_0 = np.array([1, 0, 0])
        shape = list(self._matrices.shape)
        quiver_shape = 5
        shape.append(quiver_shape)
        parameters = np.zeros(shape)
        for index in np.ndindex(self._matrices.shape):
            parameters[index][:2] = self._centers[index][:2]
            parameters[index][2:] = np.matmul(self._matrices[index], d_0)
        # TODO: this line assumes shape of parameters (can't work with halton)
        self.ax.quiver(*(parameters[:, :, i] for i in range(quiver_shape)))
        # Hide axes ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        # Add a color bar which maps values to colors.
        # plt.colorbar()
        # plt.title(title)
        plt.savefig(filename, bbox_inches="tight")
        plt.close(self.fig)


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
