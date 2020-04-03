"""
inspired by https://stackoverflow.com/questions/41955492/
how-to-plot-efficiently-a-large-number-of-3d-ellipsoids-with-matplotlib-axes3d
"""

import numpy as np
from numpy import linalg as la
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors

if __name__ == "__main__":
    from retraction_pairs import SymmetricPositiveDefinite


class ElipsoidVisualizer(object):
    def __init__(self, matrices, centers):
        self.fig = plt.figure(figsize=(8,8))
        self.ax = fig.plot(projection='3d')
        self._matrices = matrices
        self._centers = centers
        
        self._svd_matrices()
        self._normalize_singular_values()

    def _calculate_normalizer(self):
        max_radius = 0
        for radii in self._singular_values:
            if max(radii) > max_radius:
                max_radius = max(radii)

        print("Max Radius is ", max_radius)

        return max_radius

    def _svd_matrices(self):
        singular_values = np.zeros_like(self._matrices, dtype=object)
        rotations = np.zeros_like(self._matrices)
        for index in np.ndindex(self._matrices.shape)
            _, singular_values[index], rotations[index] = la.svd(self._matrices[index])

        self._singular_values = singular_values
        self._rotations = rotations

    def _process_matrix(self, center, radii, rotation)
        # calculate cartesian coordinates for the ellipsoid surface
        u = np.linspace(0.0, 2.0 * np.pi, 60)
        v = np.linspace(0.0, np.pi, 60)
        x = radii[0] * np.outer(np.cos(u), np.sin(v)) / self._normalizer
        y = radii[1] * np.outer(np.sin(u), np.sin(v)) / self._normalizer
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) / self._normalizer

        for i in range(len(x)):
            for j in range(len(x)):
                [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center
        
        self.ax.plot_surface(x, y, z,  rstride=3, cstride=3, linewidth=0.1, alpha=1, shade=True)

    def show(self):
        for index in np.ndindex(self._matrices.shape):
            self._process_matrix(
                self._centers[index], 
                self._singular_values[index],
                self._rotations[index]
                )
        
        plt.show()


def main():
    print("start")
    spd = SymmetricPositiveDefinite()
    matrices = np.zeros((3, 3), dtype=object)
    centers=zeros_like(matrices)
    for index in np.ndindex(matrices.shape):
        matrices[index] = spd.gen_point()
        centers[index] = np.array([index[0], index[1], 0])
    print("start to visualize")
    ElipsoidVisualizer(matrices, centers).show()


if __name__ == "__main__":
    main()
