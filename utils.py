import numpy as np
from numpy import linalg as la


def generate_original_function():
    def original_function(x, y):
        return (y ** 2) * (np.sin(5* (x + y)))

    return original_function


def plot_contour(ax, func, *args):
    x, y = generate_grid(*args, should_ravel=False)
    z = np.zeros(x.shape)
    for index in np.ndindex(x.shape):
        z[index] = func(x[index], y[index])
    ax.contour3D(x, y, z, 50, cmap='binary')


def generate_grid(grid_size, resolution, scale=1, should_ravel=True):
    x = np.linspace(-grid_size / resolution, grid_size / resolution, 2 * scale * grid_size)
    y = np.linspace(-grid_size / resolution, grid_size / resolution, 2 * scale * grid_size)
    x_matrix, y_matrix = np.meshgrid(x, y)
    if should_ravel:
        return x_matrix.ravel(), y_matrix.ravel()
    else:
         return x_matrix, y_matrix


def generate_kernel(rbf, scale=1):
    def kernel(x, y):
        ans = (1 / scale ** 2) * rbf(la.norm(x-y) / scale)
        return ans

    return kernel


def mse(func_a, func_b, x, y):
    errors = np.zeros(x.shape)
    for index in np.ndindex(x.shape):
        errors[index] = np.square(func_a(x[index], y[index]) - func_b(x[index], y[index]))
    return errors.mean()


def run_on_array(function, x, y):
    return np.array([function(x[index], y[index]) for index in np.ndindex(x.shape)])


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


def evaluate_original_on_points(original_function, points):
    print("started interpolation")
    x_axis, y_axis = points
    print("x shape {}".format(x_axis.shape))
    values_at_points = run_on_array(original_function, x_axis, y_axis)
    points_as_vectors = [np.array([x_0, y_0]) for x_0, y_0 in zip(x_axis, y_axis)]
    print("len: {}".format(len(points_as_vectors)))
    return points_as_vectors, values_at_points
