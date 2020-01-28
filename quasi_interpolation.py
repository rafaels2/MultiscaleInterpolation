import numpy as np
from numpy import linalg as la
from utils import generate_grid, generate_kernel, run_on_array, evaluate_original_on_points, wendland, mse


def polynom(x, y):
    return (x - 20) * (x + 20) * (y - 20) * (y + 20)


def const(x, y):
    return 1


def const_rbf(x):
    if x == 0:
        return 2
    if x == 1:
        return -0.25
    return 0


def _calculate_phi(rbf, scale, point):
    def phi(x, y):
        vector = np.array([x, y])
        return rbf(la.norm(vector - point) / scale)
    return phi


def _interpolate(original_function, points, phis):
    points_as_vectors, values_at_points = evaluate_original_on_points(original_function, points)

    def interpolant(x, y):
        return sum(phi(x, y) * value_at_point for phi, value_at_point in zip(phis, values_at_points))
    return interpolant


def quasi_scaled_interpolation(scale, original_function, grid_resolution, grid_size, rbf):
    x, y = generate_grid(grid_size, grid_resolution, scale)
    phis = [_calculate_phi(rbf, scale, np.array([x_i, y_i])) for x_i, y_i in zip(x, y)]
    return _interpolate(original_function, (x, y), phis)


def main():
    """
    This main tests the assumptions on the quasi interpolation theory
    """
    rbf = const_rbf
    x, y = generate_grid(5, 3, 1)
    phis = [_calculate_phi(rbf, 1, np.array([x_i, y_i]), polynom) for x_i, y_i in zip(x, y)]

    q = polynom
    
    def test_function(x_test, y_test):
        return sum(_calculate_phi(rbf, 1, np.array([x_i, y_i]), polynom)(x_test, y_test) for x_i, y_i in zip(x, y))

    x_test, y_test = generate_grid(5, 3, 2)

    print("mse was: {}".format(mse(test_function, q, x_test, y_test)))


if __name__ == "__main__":
    main()
