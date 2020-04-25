from datetime import datetime
import numpy.linalg as la
import pickle as pkl
import numpy as np
import time
import os

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from Config import CONFIG, DIFFS
from Tools.Utils import *
from Tools.SamplingPoints import GridParameters, Grid, symmetric_grid_params


def multiscale_interpolation(manifold, 
                             original_function,
                             grid_size,
                             resolution,
                             scaling_factor,
                             rbf,
                             number_of_scales,
                             scaled_interpolation_method,
                             is_approximating_on_tangent):
    f_j = manifold.zero_func
    e_j = act_on_functions(manifold.log, f_j, original_function)
    for scale_index in range(1, number_of_scales + 1):
        scale = scaling_factor ** scale_index
        print("NEW SCALE: {}".format(scale))

        if is_approximating_on_tangent:
            function_to_interpolate = e_j
        else:
            function_to_interpolate = act_on_functions(manifold.exp, manifold.zero_func, e_j)

        current_grid_parameters = [
            ('Grid', symmetric_grid_params(grid_size + 1, scale / resolution)),
            # Can add here more grids (borders)
        ]

        s_j = scaled_interpolation_method(
            manifold,
            function_to_interpolate,
            current_grid_parameters,
            rbf,
            scale,
            is_approximating_on_tangent).approximation
        print("interpolated!")

        if is_approximating_on_tangent:
            function_added_to_f_j = s_j
        else:
            function_added_to_f_j = act_on_functions(manifold.log, manifold.zero_func, s_j)

        f_j = act_on_functions(manifold.exp, f_j, function_added_to_f_j)
        e_j = act_on_functions(manifold.log, f_j, original_function)

    return f_j


def calculate_max_derivative(original_function, grid_params, m):
    def derivative(x, y):
        m = grid_params.mesh_norm
        vals = [original_function(x+m, y+m),
         original_function(x+m, y-m),
         original_function(x-m, y+m),
         original_function(x-m, y-m)]

         xy = original_function(x, y)

        return max([m.distance(val - xy) for val in vals])

    return Grid(1, derivative, grid_params).evaluation


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
                  "deriveatives.png",
                  norm_visualization=True)

    with set_output_directory(experiment_name):
        with open("config.pkl", "wb") as f:
            pkl.dump(config, f)

        interpolant = multiscale_interpolation(
            manifold,
            number_of_scales=number_of_scales,
            original_function=original_function,
            grid_size = grid_size,
            resolution=base_resolution,
            scaling_factor=scaling_factor,
            rbf=rbf,
            scaled_interpolation_method=scaled_interpolation_method,
            is_approximating_on_tangent=is_approximating_on_tangent
        )

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

    return mse, interpolant


def run_all_experiments(config, diffs, *args):
    mses = dict()
    calculation_time = list()
    interpolants = list()
    execution_name = config["EXECUTION_NAME"]
    path = "{}_{}".format(execution_name, time.strftime("%Y%m%d__%H%M%S"))
    with set_output_directory(path):
        for diff in diffs:
            t_0 = datetime.now()
            current_config = config.copy()
            for k, v in diff.items():
                current_config[k] = v
            mse, interpolant = run_single_experiment(current_config, *args)
            mse = np.log(mse)
            mse_label = current_config["MSE_LABEL"]
            current_mses = mses.get(mse_label, list())
            current_mses.append(mse)
            mses[mse_label] = current_mses
            t_f = datetime.now()
            calculation_time.append(t_f - t_0)
            interpolants.append(interpolant)
    
        plot_lines(mses, "mses.png", "Error in different runs", "Iteration", "log(Error)")

    print("MSEs are: {}".format(mses))
    print("times are: {}".format(calculation_time))
    return mses, interpolants


def main():
    rbf = wendland
    original_function = CONFIG["ORIGINAL_FUNCTION"]
    config = CONFIG
    scaling_factor = CONFIG["SCALING_FACTOR"]
    output_dir = CONFIG["OUTPUT_DIR"]
    diffs = DIFFS

    with set_output_directory(output_dir):
        results,interpolants = run_all_experiments(config, diffs, rbf, original_function)

    return results, interpolants


if __name__ == "__main__":
    _, interpolants = main()
