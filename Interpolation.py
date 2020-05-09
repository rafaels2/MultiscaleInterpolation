from datetime import datetime
import numpy.linalg as la
import pickle as pkl
import numpy as np
import time
import os
import abc

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from Config import CONFIG, DIFFS
from Tools.Utils import *
from Tools.SamplingPoints import GridParameters, Grid, symmetric_grid_params


def run_single_experiment(config, rbf, original_function):
    grid_size = config["GRID_SIZE"]
    base_resolution = config["BASE_RESOLUTION"]
    plot_resolution_factor = config["PLOT_RESOLUTION_FACTOR"]
    scale = config["SCALE"]
    number_of_scales = config["NUMBER_OF_SCALES"]
    test_mesh_norm = config["TEST_MESH_NORM"]
    scaling_factor = config["SCALING_FACTOR"]
    experiment_name = config["NAME"] or "temp"
    manifold = config["MANIFOLD"]
    scaled_interpolation_method=config["SCALED_INTERPOLATION_METHOD"]
    norm_visualization = config["NORM_VISUALIZATION"]
    is_approximating_on_tangent = config["IS_APPROXIMATING_ON_TANGENT"]
    is_adaptive = config["IS_ADAPTIVE"]
    multiscale_method = config["MULTISCALE_METHOD"]

    grid_params = symmetric_grid_params(grid_size, test_mesh_norm)
    true_values_on_grid = Grid(1, original_function, grid_params).evaluation

    manifold.plot(
        true_values_on_grid,
        "original",
        "original.png",
        norm_visualization=norm_visualization
    )

    plot_and_save(calculate_max_derivative(original_function, grid_params, manifold),
                  "max derivatives",
                  "deriveatives.png")

    for i, interpolant in enumerate(multiscale_method(
             manifold,
             original_function,
             grid_size,
             resolution,
             rbf,
             number_of_scales,
             scaled_interpolation_method).run_all_scales()):    
        with set_output_directory("{}_{}".format(experiment_name, i+1)):
            with open("config.pkl", "wb") as f:
                pkl.dump(config, f)

            approximated_values_on_grid = Grid(1, interpolant, grid_params).evaluation

            manifold.plot(
                approximated_values_on_grid,
                "approximation",
                "approximation.png",
                norm_visualization=norm_visualization
            )

            error = manifold.calculate_error(approximated_values_on_grid, true_values_on_grid)
            plot_and_save(error, "difference map", "difference.png")
            
            mse = np.mean(error)
            with open("results.pkl", "wb") as f:
                results = {
                    "original_values": true_values_on_grid,
                    "approximation": approximated_values_on_grid,
                    "errors": error,
                    "mse": mse,
                }
                pkl.dump(results, f)

        yield mse


def run_all_experiments(config, diffs, *args):
    mses = dict()
    calculation_time = list()
    interpolants = list()
    execution_name = config["EXECUTION_NAME"]
    path = "{}_{}".format(execution_name, time.strftime("%Y%m%d__%H%M%S"))
    with set_output_directory(path):
        for diff in diffs:
            current_config = config.copy()

            for k, v in diff.items():
                current_config[k] = v

            t_0 = datetime.now()

            for mse in run_single_experiment(current_config, *args):
                t_f = datetime.now()
                calculation_time.append(t_f - t_0)
                mse = np.log(mse)
                mse_label = current_config["MSE_LABEL"]
                current_mses = mses.get(mse_label, list())
                current_mses.append(mse)
                mses[mse_label] = current_mses
                t_0 = datetime.now()
    
        plot_lines(mses, "mses.png", "Error in different runs", "Iteration", "log(Error)")

    print("MSEs are: {}".format(mses))
    print("times are: {}".format(calculation_time))
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

    return results, interpolants


if __name__ == "__main__":
    _, interpolants = main()
