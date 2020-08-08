from collections import namedtuple

import numpy as np
from scipy.spatial.transform import Rotation

from Tools.Visualization import RotationVisualizer
from .RigidRotations import RigidRotations


TestPoint = namedtuple('TestPoint', ['x', 'y', 'w_x', 'w_y', 'w_z'])


def generate_triangle(levels, scale):
    a = scale * np.array([0, np.sqrt(0.75)])
    b = scale * np.array([-0.5, 0])
    c = scale * np.array([0.5, 0])

    for i in range(levels):
        for j in range(levels):
            for k in range(levels):
                normalizer = i + j + k
                if normalizer == 0:
                    continue
                p = (i * a + j * b + k * c) / normalizer
                yield TestPoint(p[0], p[1], i/normalizer, j/normalizer, k/normalizer)


def average_test():
    x_rot = (Rotation.from_euler('x', 0.5)).as_matrix()
    y_rot = (Rotation.from_euler('y', 0.5)).as_matrix()
    z_rot = (Rotation.from_euler('z', 0.5)).as_matrix()

    m = RigidRotations()
    triangles = list(generate_triangle(4, 8))
    matrices = np.zeros(len(triangles), dtype=object)
    centers = np.zeros(len(triangles), dtype=object)

    for i, p in enumerate(triangles):
        print("Now: ", i / len(triangles))
        matrices[i] = (m.average([x_rot, y_rot, z_rot], [p.w_x, p.w_y, p.w_z]))
        centers[i] = (np.array([p.x, p.y, 0]))

    visualizer = RotationVisualizer(matrices, centers)
    visualizer.save("average_results_0_5.png", "Averages")


def main():
    average_test()


if __name__ == '__main__':
    main()
