from datetime import datetime
import numpy.linalg as la
import pickle as pkl
import numpy as np
import time
import os

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from ApproximationMethods.NoNormalization import normalization_cache
from Config import CONFIG, DIFFS
from Manifolds.RealNumbers import Calibration
from Tools.GridUtils import calculate_max_derivative
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
                             is_approximating_on_tangent,
                             is_adaptive):
    f_j = manifold.zero_func
    e_j = act_on_functions(manifold.log, f_j, original_function)
    for scale_index in range(1, number_of_scales + 1):
        scale = scaling_factor ** scale_index
        print("NEW SCALE: {}".format(scale))

        if is_approximating_on_tangent:
            function_to_interpolate = e_j
        elif is_adaptive:
            function_to_interpolate = (e_j, act_on_functions(manifold.exp, manifold.zero_func, e_j))
        else:
            function_to_interpolate = act_on_functions(manifold.exp, manifold.zero_func, e_j)

        current_grid_parameters = [
            ('Grid', symmetric_grid_params(grid_size + 1, scale / resolution)),
            # Can add here more grids (borders)
        ]

        method = scaled_interpolation_method(
            manifold,
            function_to_interpolate,
            current_grid_parameters,
            rbf,
            scale,
            is_approximating_on_tangent)

        s_j = method.approximation
        print("interpolated!")

        if is_approximating_on_tangent or is_adaptive:
            function_added_to_f_j = s_j
        else:
            function_added_to_f_j = act_on_functions(manifold.log, manifold.zero_func, s_j)

        f_j = act_on_functions(manifold.exp, f_j, function_added_to_f_j)
        e_j = act_on_functions(manifold.log, f_j, original_function)
        yield scale / resolution, f_j


def run_single_experiment(config, original_function):
    rbf = config["RBF"]
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

    for i, (mesh_norm, interpolant) in enumerate(multiscale_interpolation(
            manifold,
            number_of_scales=number_of_scales,
            original_function=original_function,
            grid_size = grid_size,
            resolution=base_resolution,
            scaling_factor=scaling_factor,
            rbf=rbf,
            scaled_interpolation_method=scaled_interpolation_method,
            is_approximating_on_tangent=is_approximating_on_tangent,
            is_adaptive=is_adaptive
            )):    
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

            if config["ERROR_CALC"]:
                mse = np.average(error)
            else:
                mse = la.norm(error)
            with open("results.pkl", "wb") as f:
                results = {
                    "original_values": true_values_on_grid,
                    "approximation": approximated_values_on_grid,
                    "errors": error,
                    "mse": mse,
                    "mesh_norm": mesh_norm
                }
                pkl.dump(results, f)

        yield mse, mesh_norm, error


def run_all_experiments(config, diffs, *args):
    mses = dict()
    mesh_norms = dict()
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

            for mse, mesh_norm, _ in run_single_experiment(current_config, *args):
                t_f = datetime.now()
                calculation_time.append(t_f - t_0)
                if not current_config["ERROR_CALC"]:
                    mse = np.log(mse)
                mse_label = current_config["MSE_LABEL"]
                current_mses = mses.get(mse_label, list())
                current_mesh_norms = mesh_norms.get(mse_label, list())
                current_mses.append(mse)
                current_mesh_norms.append(np.log(mesh_norm))
                mses[mse_label] = current_mses
                mesh_norms[mse_label] = current_mesh_norms
                t_0 = datetime.now()
    
        plot_lines(mesh_norms, mses, "mses.svg", "Error in different runs", "log(h_x)", "log(Error)")

    print("MSEs are: {}".format(mses))
    print("mesh_norms are: {}".format(mesh_norms))
    print("times are: {}".format(calculation_time))
    return mses


def calibrate(config, diffs, *args):
    mses = dict()
    mesh_norms = dict()
    execution_name = config["EXECUTION_NAME"]
    path = "{}_{}".format(execution_name, time.strftime("%Y%m%d__%H%M%S"))
    with set_output_directory(path):
        for diff in diffs:
            current_config = config.copy()

            for k, v in diff.items():
                current_config[k] = v

            current_config["MANIFOLD"] = Calibration()

            for _, mesh_norm, error in run_single_experiment(current_config, *args):
                mse = np.average(error)
                mse_label = current_config["MSE_LABEL"]
                current_mses = mses.get(mse_label, list())
                current_mesh_norms = mesh_norms.get(mse_label, list())
                current_mses.append(mse)
                current_mesh_norms.append(mesh_norm)
                mses[mse_label] = current_mses
                mesh_norms[mse_label] = current_mesh_norms
                normalization_cache[
                    (current_config["RBF"].__name__, mesh_norm, mesh_norm * current_config["BASE_RESOLUTION"])
                ] = mse

        plot_lines(mesh_norms, mses, "mses.svg", "Error in different runs", "log(h_x)", "log(Error)")

    print("MSEs are: {}".format(mses))
    print("mesh_norms are: {}".format(mesh_norms))
    return mses


def main():
    rbf = wendland_3_0
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
