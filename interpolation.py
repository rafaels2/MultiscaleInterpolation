"""
Issues:
1. should be scaled or have more points
"""

import numpy as np
import numpy.linalg as la
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

GRID_SIZE = 4
ORIGINAL_SCALE = 2
BASE_RESOLUTION = 3
PLOT_RESOLUTION_FACTOR = 4
DIMENSION = 2
SCALE = 2
NUMBER_OF_SCALES = 1


def sum_functions(a, b):
    def new_func(*args):
        return a(*args) + b(*args)

    return new_func


def sub_functions(a, b):
    def new_func(*args):
        return a(*args) - b(*args)

    return new_func


def zero_func(*args):
    return 0


def wendland(x):
    if x < 0:
        raise ValueError("x should be > 0, not {}".format(x))
    if x > 1:
        return 0
    else:
        return (1 + (4 * x)) * ((1 - x) ** 4)


def run_on_array(function, x, y):
    return np.array([function(x[index], y[index]) for index in np.ndindex(x.shape)])


def interpolate(phi, original_function, points):
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


def generate_kernel(rbf, scale=1):
    def kernel(x, y):
        # TODO: Maybe scale should be squared?
        ans = (1 / scale) * rbf(la.norm(x-y) / scale)
        return ans

    return kernel


def generate_original_function():
    def original_function(x, y):
        return y * (np.sin(x) - 1)

    return original_function


def mse(func_a, func_b, x, y):
    errors = np.zeros(x.shape)
    for index in np.ndindex(x.shape):
        errors[index] = np.square(func_a(x[index], y[index]) - func_b(x[index], y[index]))
    return errors.mean()


def generate_grid(grid_size, resolution, scale=1, should_ravel=True):
    x = np.linspace(-grid_size / resolution, grid_size / resolution, 2 * scale * grid_size)
    y = np.linspace(-grid_size / resolution, grid_size / resolution, 2 * scale * grid_size)
    x_matrix, y_matrix = np.meshgrid(x, y)
    if should_ravel:
        return x_matrix.ravel(), y_matrix.ravel()
    else:
         return x_matrix, y_matrix


def plot_contour(ax, func, *args):
    import ipdb; ipdb.set_trace()
    x, y = generate_grid(*args, should_ravel=False)
    z = np.zeros(x.shape)
    for index in np.ndindex(x.shape):
        z[index] = func(x[index], y[index])
    ax.contour3D(x, y, z, 50, cmap='binary')


def scaled_interpolation(scale, original_function, grid_resolution, grid_size, rbf):
    x, y = generate_grid(grid_size, grid_resolution, scale)
    phi = generate_kernel(rbf, scale)
    return interpolate(phi, original_function, (x, y))


def multiscale_interpolation(number_of_scales, original_function, **kwargs):
    f_j = zero_func
    e_j = original_function
    for scale in range(1, number_of_scales + 1):
        print("NEW SCALE: {}".format(scale))
        s_j = scaled_interpolation(scale, e_j, **kwargs.copy())
        print("interpolated!")
        f_j = sum_functions(f_j, s_j)
        e_j = sub_functions(e_j, s_j)

    return f_j


def main():
    rbf = wendland
    original_function = generate_original_function()

    plt.figure()
    ax = plt.axes(projection='3d')
    plot_contour(ax, original_function, GRID_SIZE, BASE_RESOLUTION * PLOT_RESOLUTION_FACTOR, SCALE)
    plt.show()


    interpolant = multiscale_interpolation(
        number_of_scales=NUMBER_OF_SCALES,
        original_function=original_function,
        grid_resolution=BASE_RESOLUTION,
        grid_size=GRID_SIZE,
        rbf=rbf
    )

    plt.figure()
    ax = plt.axes(projection='3d')
    plot_contour(ax, interpolant, GRID_SIZE, BASE_RESOLUTION * PLOT_RESOLUTION_FACTOR, SCALE)
    plt.show()

    test_x, test_y = generate_grid(GRID_SIZE, BASE_RESOLUTION * PLOT_RESOLUTION_FACTOR, SCALE)
    print("MSE was: ", mse(original_function, interpolant, test_x, test_y))


if __name__ == "__main__":
    main()


def scaled_main():
    rbf = wendland
    original_function = generate_original_function()

    plt.figure()
    ax = plt.axes(projection='3d')
    plot_contour(ax, original_function, GRID_SIZE)
    plt.show()

    interpolant = scaled_interpolation(
        scale=SCALE,
        original_function=original_function,
        grid_resolution=BASE_RESOLUTION,
        grid_size=GRID_SIZE,
        rbf=rbf
    )
    
    plt.figure()
    ax = plt.axes(projection='3d')
    plot_contour(ax, interpolant, GRID_SIZE)
    plt.show()

    test_x, test_y = generate_grid(GRID_SIZE, BASE_RESOLUTION * PLOT_RESOLUTION_FACTOR ,SCALE)
    print("MSE was: ", mse(original_function, interpolant, test_x, test_y))


def nonscaled_interpolation_main():
    rbf = wendland
    # Generate grid
    x, y = generate_grid(GRID_SIZE, PLOT_RESOLUTION_FACTOR)

    # fine_grid = generate_grid(GRID_SIZE, PLOT_RESOLUTION_FACTOR)

    # Create phi
    phi = generate_kernel(rbf)

    # Get original function
    original_function = generate_original_function()

    # Interpolate
    interpolant = interpolate(phi, original_function, (x, y))
    # fine_grid_interpolated_values = map(interpolant, fine_grid)

    # Plot values on finer grid
    plt.figure()
    ax = plt.axes(projection='3d')
    plot_contour(ax, original_function, GRID_SIZE)
    plt.show()
    plt.figure()
    ax = plt.axes(projection='3d')
    plot_contour(ax, interpolant, GRID_SIZE)
    plt.show()
    x = np.linspace(-GRID_SIZE, GRID_SIZE, 8 * GRID_SIZE)
    y = np.linspace(-GRID_SIZE, GRID_SIZE, 8 * GRID_SIZE)
    print("MSE was: ", mse(original_function, interpolant, x, y))
