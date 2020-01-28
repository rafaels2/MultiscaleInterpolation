import numpy as np
from numpy import linalg as la
from utils import generate_grid, generate_kernel, run_on_array


def _interpolate(phi, original_function, points):
    """
    Generating I_Xf(x) for the given kernel and points
    :param phi: Kernel
    :param original_function:
    :param points:
    :return:
    """
    print("started interpolation")
    x_axis, y_axis = points
    print("x shape {}".format(x_axis.shape))
    values_at_points = run_on_array(original_function, x_axis, y_axis)
    points_as_vectors = [np.array([x_0, y_0]) for x_0, y_0 in zip(x_axis, y_axis)]
    print("len: {}".format(len(points_as_vectors)))
    kernel = np.array([[phi(x_i, x_j) for x_j in points_as_vectors] for x_i in points_as_vectors])
    coefficients = np.matmul(la.inv(kernel), values_at_points)
    print(kernel)

    def interpolant(x, y):
        return sum(b_j * phi(np.array([x, y]), x_j)
                   for b_j, x_j in zip(coefficients, points_as_vectors))
    return interpolant


def naive_scaled_interpolation(scale, original_function, grid_resolution, grid_size, rbf):
    x, y = generate_grid(grid_size, grid_resolution, scale)
    phi = generate_kernel(rbf, scale)
    return _interpolate(phi, original_function, (x, y))