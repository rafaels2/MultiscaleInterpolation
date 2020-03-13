from datetime import datetime
import numpy.linalg as la
import pickle as pkl
import numpy as np
import os

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from quasi_interpolation import quasi_scaled_interpolation
from naive import naive_scaled_interpolation
from config import CONFIG
from utils import *


def multiscale_interpolation(number_of_scales, original_function, scaling_factor,
    scaled_interpolation_method=naive_scaled_interpolation, **kwargs):
    f_j = zero_func
    e_j = original_function
    for scale_index in range(1, number_of_scales + 1):
        scale = scaling_factor ** scale_index
        print("NEW SCALE: {}".format(scale))
        s_j = scaled_interpolation_method(scale, e_j, **kwargs.copy())
        print("interpolated!")
        f_j = sum_functions(f_j, s_j)
        e_j = sub_functions(e_j, s_j)

    return f_j


def run_single_experiment(config, rbf, original_function):
    grid_size = config["GRID_SIZE"]
    base_resolution = config["BASE_RESOLUTION"]
    plot_resolution_factor = config["PLOT_RESOLUTION_FACTOR"]
    scale = config["SCALE"]
    number_of_scales = config["NUMBER_OF_SCALES"]
    test_scale = config["TEST_SCALE"]
    scaling_factor = config["SCALING_FACTOR"]
    experiment_name = config["NAME"] or "temp"

    with set_output_directory(experiment_name):
        with open("config.pkl", "wb") as f:
            pkl.dump(config, f)

        plt.figure()
        plt.title("original")
        ax = plt.axes(projection='3d')
        true_values_on_grid = plot_contour(ax, original_function, grid_size, base_resolution, test_scale)
        plt.savefig("original.png")

        interpolant = multiscale_interpolation(
            number_of_scales=number_of_scales,
            original_function=original_function,
            grid_resolution=base_resolution,
            grid_size=grid_size,
            scaling_factor=scaling_factor,
            rbf=rbf,
            scaled_interpolation_method=quasi_scaled_interpolation
        )
        
        plt.figure()
        plt.title("approximation")
        ax = plt.axes(projection='3d')
        approximated_values_on_grid = plot_contour(ax, interpolant, grid_size, base_resolution, test_scale)
        plt.savefig("approximation.png")

        plt.figure()
        plt.title("difference map")
        plt.imshow(approximated_values_on_grid - true_values_on_grid)
        cb = plt.colorbar()
        plt.savefig("difference.png")
        cb.remove()
        
        mse = np.mean(np.square(approximated_values_on_grid - true_values_on_grid))
        with open("log.dat", "w") as f:
            f.write("MSE was: {}".format(mse))

    return mse


def run_all_experiments(config, diffs, *args):
    mses = list()
    execution_name = config["EXECUTION_NAME"]
    path = "{}_{}".format(execution_name, str(datetime.now()))
    with set_output_directory(path):
        for diff in diffs:
            current_config = config.copy()
            for k, v in diff.items():
                current_config[k] = v
            mse = run_single_experiment(current_config, *args)
            mses.append(mse)

    print("MSEs are: {}".format(mses))
    return mses


def main():
    rbf = wendland
    original_function = generate_original_function()
    config = CONFIG
    scaling_factor = CONFIG["SCALING_FACTOR"]
    output_dir = CONFIG["OUTPUT_DIR"]

    diffs = [
        {"NAME": "{}_scale".format(x), "NUMBER_OF_SCALES": x} for x in range(1, 5)
    ] + [
        {"NAME": "single_scale_{}".format(i), 
         "NUMBER_OF_SCALES": 1,
         "SCALING_FACTOR": scaling_factor ** i} for i in range(1, 5)
    ]

    with set_output_directory(output_dir):
        results = run_all_experiments(config, diffs, rbf, original_function)

    return results


if __name__ == "__main__":
    main()
