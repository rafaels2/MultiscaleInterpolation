import numpy as np
import numpy.linalg as la


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
    values_at_points = (original_function(x_j) for x_j in points)
    kernel = np.array([[phi(x_i, x_j) for x_j in points] for x_i in points])
    coefficients = la.inv(kernel) * values_at_points

    def interpolant(x):
        return sum(b_j * phi(x, x_j) for b_j, x_j in zip(coefficients, points))

    return interpolant


def main():
    # Generate grid
    # Create phi
    # Get original function
    # Calculate values on all points
    # Interpolate
    # Plot values on finer grid
    pass


if __name__ == "__main__":
    main()
