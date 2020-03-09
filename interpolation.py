import numpy as np
import numpy.linalg as la
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from utils import plot_contour, generate_grid, mse, \
    sum_functions, sub_functions, zero_func, wendland, generate_original_function
from naive import naive_scaled_interpolation
from quasi_interpolation import quasi_scaled_interpolation
from config import CONFIG

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
    
    grid_size = CONFIG["GRID_SIZE"]
    base_resolution = CONFIG["BASE_RESOLUTION"]
    plot_resolution_factor = CONFIG["PLOT_RESOLUTION_FACTOR"]
    scale = CONFIG["SCALE"]
    number_of_scales = CONFIG["NUMBER_OF_SCALES"]
    test_scale = CONFIG["TEST_SCALE"]

    plt.figure()
    ax = plt.axes(projection='3d')
    true_values_on_grid = plot_contour(ax, original_function, grid_size, base_resolution * plot_resolution_factor, test_scale)
    plt.show()

    interpolant = multiscale_interpolation(
        number_of_scales=number_of_scales,
        original_function=original_function,
        grid_resolution=base_resolution,
        grid_size=grid_size,
        rbf=rbf,
        scaled_interpolation_method=quasi_scaled_interpolation
    )
    
    plt.figure()
    ax = plt.axes(projection='3d')
    approximated_values_on_grid = plot_contour(ax, interpolant, grid_size, base_resolution * plot_resolution_factor, test_scale)
    plt.show()

    print("MSE was: ", np.mean(np.square(approximated_values_on_grid - true_values_on_grid)))


if __name__ == "__main__":
    main()
