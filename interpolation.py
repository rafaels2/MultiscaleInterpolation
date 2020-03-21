import numpy.linalg as la
import pickle as pkl
import numpy as np
import time
import os

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from config import CONFIG, DIFFS
from utils import *


def multiscale_interpolation(manifold, number_of_scales, original_function, scaling_factor,
    scaled_interpolation_method, **kwargs):
    f_j = manifold.zero_func
    e_j = act_on_functions(manifold.log, f_j, original_function)
    for scale_index in range(1, number_of_scales + 1):
        scale = scaling_factor ** scale_index
        print("NEW SCALE: {}".format(scale))
        s_j = scaled_interpolation_method(manifold, scale, e_j, **kwargs.copy())
        print("interpolated!")
        f_j = act_on_functions(manifold.exp, f_j, s_j)
        e_j = act_on_functions(manifold.log, f_j, original_function)

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
    manifold = config["MANIFOLD"]
    scaled_interpolation_method=config["SCALED_INTERPOLATION_METHOD"]

    with set_output_directory(experiment_name):
        with open("config.pkl", "wb") as f:
            pkl.dump(config, f)

        true_values_on_grid = evaluate_on_grid(original_function, grid_size, base_resolution, test_scale, should_log=True)
        manifold.plot(true_values_on_grid, "original", "original.png")

        interpolant = multiscale_interpolation(
            manifold,
            number_of_scales=number_of_scales,
            original_function=original_function,
            grid_resolution=base_resolution,
            grid_size=grid_size,
            scaling_factor=scaling_factor,
            rbf=rbf,
            scaled_interpolation_method=scaled_interpolation_method
        )

        approximated_values_on_grid = evaluate_on_grid(interpolant, grid_size, base_resolution, test_scale, should_log=True)
        manifold.plot(approximated_values_on_grid, "approximation", "approximation.png")

        error = manifold.calculate_error(approximated_values_on_grid, true_values_on_grid)
        plot_and_save(error, "difference map", "difference.png")
        
        mse = np.mean(error)
        with open("results.pkl", "wb") as f:
            results = {
                "original_values": true_values_on_grid,
                "approximation": approximated_values_on_grid,
                "errors": error,
                "mse": mse
            }
            pkl.dump(results, f)

    return mse


def run_all_experiments(config, diffs, *args):
    mses = list()
    execution_name = config["EXECUTION_NAME"]
    path = "{}_{}".format(execution_name, time.strftime("%Y%m%d__%H%M%S"))
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
    original_function = CONFIG["ORIGINAL_FUNCTION"]
    config = CONFIG
    scaling_factor = CONFIG["SCALING_FACTOR"]
    output_dir = CONFIG["OUTPUT_DIR"]
    diffs = DIFFS

    with set_output_directory(output_dir):
        results = run_all_experiments(config, diffs, rbf, original_function)

    return results


if __name__ == "__main__":
    main()
