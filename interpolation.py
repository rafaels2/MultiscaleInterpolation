"""
Issues:
1. should be scaled or have more points
"""

import numpy as np
import numpy.linalg as la
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from utils import plot_contour, generate_grid, mse, run_on_array, \
    sum_functions, sub_functions, zero_func, wendland, generate_original_function
from naive import naive_scaled_interpolation
from quasi_interpolation import quasi_scaled_interpolation

GRID_SIZE = 4
ORIGINAL_SCALE = 2
BASE_RESOLUTION = 3
PLOT_RESOLUTION_FACTOR = 4
DIMENSION = 2
SCALE = 2
NUMBER_OF_SCALES = 3


def multiscale_interpolation(number_of_scales, original_function, scaled_interpolation_method=naive_scaled_interpolation, **kwargs):
    f_j = zero_func
    e_j = original_function
    for scale in range(1, number_of_scales + 1):
        print("NEW SCALE: {}".format(scale))
        s_j = scaled_interpolation_method(scale, e_j, **kwargs.copy())
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
        rbf=rbf,
        scaled_interpolation_method=quasi_scaled_interpolation
    )

    plt.figure()
    ax = plt.axes(projection='3d')
    plot_contour(ax, interpolant, GRID_SIZE, BASE_RESOLUTION * PLOT_RESOLUTION_FACTOR, SCALE)
    plt.show()

    test_x, test_y = generate_grid(GRID_SIZE, BASE_RESOLUTION * PLOT_RESOLUTION_FACTOR, SCALE)
    print("MSE was: ", mse(original_function, interpolant, test_x, test_y))


if __name__ == "__main__":
    main()
