"""
Deprecated
"""
import numpy as np
from numpy import linalg as la

from DataSites.GridUtils import evaluate_on_grid
from Tools.Utils import generate_kernel


def _interpolate(phi, original_function, points):
    """
    Generating I_Xf(x) for the given kernel and points
    :param phi: Kernel
    :param original_function:
    :param points:
    :return:
    """
    points_as_vectors, values_at_points = evaluate_on_grid(
        original_function, points=points
    )
    kernel = np.array(
        [[phi(x_i, x_j) for x_j in points_as_vectors] for x_i in points_as_vectors]
    )
    coefficients = np.matmul(la.inv(kernel), values_at_points)
    print(kernel)

    def interpolant(x, y):
        return sum(
            b_j * phi(np.array([x, y]), x_j)
            for b_j, x_j in zip(coefficients, points_as_vectors)
        )

    return interpolant


def naive_scaled_interpolation(
    scale, original_function, grid_resolution, grid_size, rbf
):
    # Warning: deprecated
    print("Warning: deprecated code")
    x, y = generate_grid(grid_size, grid_resolution, scale)
    phi = generate_kernel(rbf, scale)
    return _interpolate(phi, original_function, (x, y))


def generate_grid(grid_size, resolution, scale=1, should_ravel=True):
    print("creating a grid", 2 * resolution / scale)
    y = np.linspace(-grid_size, grid_size, int(2 * resolution / scale))
    x = np.linspace(-grid_size, grid_size, int(2 * resolution / scale))
    x_matrix, y_matrix = np.meshgrid(x, y)
    if should_ravel:
        return x_matrix.ravel(), y_matrix.ravel()
    else:
        return x_matrix, y_matrix
