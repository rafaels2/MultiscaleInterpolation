import numpy as np
import numpy.linalg as la
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

GRID_SIZE = 10
PLOT_RESOLUTION_FACTOR = 4


def wendland(x):
    if x < 0:
        raise ValueError("x should be > 0, not {}".format(x))
    if x > 1:
        return 0
    else:
        return (1 + (4 * x)) * ((1 - x) ** 4)


def interpolate(phi, original_function, points):
    """
    Generating I_Xf(x) for the given kernel and points
    :param phi: Kernel
    :param original_function:
    :param points:
    :return:
    """
    values_at_points = original_function(*points)
    points_as_vectors = [np.array([x, y]) for x, y in zip(*points)]
    kernel = np.array([[phi(x_i, x_j) for x_j in points_as_vectors] for x_i in points_as_vectors])
    coefficients = np.matmul(la.inv(kernel), values_at_points)

    def interpolant(x, y):
        return sum(b_j * phi(np.array([x, y]), x_j)
                   for b_j, x_j in zip(coefficients, points_as_vectors))
    return interpolant


def generate_kernel(rbf):
    def kernel(x, y):
        return rbf(la.norm(x-y))

    return kernel


def generate_original_function():
    def original_function(x, y):
        return x*y

    return original_function


def mse(func_a, func_b, x, y):
    errors = np.zeros(x.shape)
    for index in np.ndindex(x.shape):
        errors[index] = np.square(func_a(x[index], y[index]) - func_b(x[index], y[index]))
    return errors.mean()


def plot_contour(ax, func, grid_size):
    x = np.linspace(-grid_size, grid_size, 8 * grid_size)
    y = np.linspace(-grid_size, grid_size, 8 * grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for index in np.ndindex(X.shape):
        Z[index] = func(X[index], Y[index])
    ax.contour3D(X, Y, Z, 50, cmap='binary')


def main():
    rbf = wendland

    # Generate grid
    x = np.linspace(-GRID_SIZE, GRID_SIZE, 2 * GRID_SIZE)
    y = np.linspace(-GRID_SIZE, GRID_SIZE, 2 * GRID_SIZE)
    grid = (x, y)
    # fine_grid = generate_grid(GRID_SIZE, PLOT_RESOLUTION_FACTOR)

    # Create phi
    phi = generate_kernel(rbf)

    # Get original function
    original_function = generate_original_function()

    # Calculate values on all points
    # fine_grid_values = map(original_function, fine_grid)

    # Interpolate
    interpolant = interpolate(phi, original_function, grid)
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


if __name__ == "__main__":
    main()
