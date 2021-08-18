from datetime import datetime
import pickle as pkl
import numpy as np
import time

from Config.Config import config
from Config.Options import options
from DataSites.Generation.Grid import get_grid
from DataSites.GridUtils import calculate_max_derivative
from DataSites.Storage.Grid import Grid
from Tools.Utils import *
from DataSites.GridUtils import symmetric_grid_params

config_plt(plt)


def multiscale_interpolation():
    f_j = config.MANIFOLD.zero_func
    e_j = act_on_functions(config.MANIFOLD.log, f_j, config.ORIGINAL_FUNCTION)
    for scale_index in range(1, config.NUMBER_OF_SCALES + 1):
        scale = config.SCALING_FACTOR ** scale_index
        print("NEW SCALE: {}".format(scale))

        if config.IS_APPROXIMATING_ON_TANGENT:
            function_to_interpolate = e_j
        elif config.IS_ADAPTIVE:
            function_to_interpolate = (
                e_j,
                act_on_functions(config.MANIFOLD.exp, config.MANIFOLD.zero_func, e_j),
            )
        else:
            function_to_interpolate = act_on_functions(
                config.MANIFOLD.exp, config.MANIFOLD.zero_func, e_j
            )

        fill_distance = scale / config.BASE_RESOLUTION
        current_grid_parameters = symmetric_grid_params(
            config.GRID_SIZE + 0.5, fill_distance
        )

        approximation_method = options.get_option(
            "approximation_method", config.SCALED_INTERPOLATION_METHOD
        )(
            function_to_interpolate,
            current_grid_parameters,
            scale,
        )
        s_j = approximation_method.approximation

        if config.IS_APPROXIMATING_ON_TANGENT or config.IS_ADAPTIVE:
            function_added_to_f_j = s_j
        else:
            function_added_to_f_j = act_on_functions(
                config.MANIFOLD.log, config.MANIFOLD.zero_func, s_j
            )

        f_j = act_on_functions(config.MANIFOLD.exp, f_j, function_added_to_f_j)
        e_j = act_on_functions(config.MANIFOLD.log, f_j, config.ORIGINAL_FUNCTION)
        yield fill_distance, f_j


def run_single_experiment():
    grid_params = symmetric_grid_params(config.GRID_SIZE, config.TEST_MESH_NORM)
    sites = get_grid(*grid_params)
    true_values_on_grid = Grid(
        sites, 1, config.ORIGINAL_FUNCTION, grid_params.fill_distance
    ).evaluation

    config.MANIFOLD.plot(
        true_values_on_grid,
        "Original",
        "original.png",
        norm_visualization=config.NORM_VISUALIZATION,
    )

    plot_and_save(
        calculate_max_derivative(
            config.ORIGINAL_FUNCTION, grid_params, config.MANIFOLD
        ),
        "Max Derivatives",
        "derivatives.png",
    )

    for i, (fill_distance, interpolant) in enumerate(multiscale_interpolation()):
        with set_output_directory("{}_{}".format(config.NAME, i + 1)):
            with open("config.pkl", "wb") as f:
                pkl.dump(config, f)

            sites = get_grid(*grid_params)
            approximated_values_on_grid = Grid(
                sites, 1, interpolant, grid_params.fill_distance
            ).evaluation

            config.MANIFOLD.plot(
                approximated_values_on_grid,
                "Approximation",
                "approximation.png",
                norm_visualization=config.NORM_VISUALIZATION,
            )

            error = config.MANIFOLD.calculate_error(
                approximated_values_on_grid, true_values_on_grid
            )
            plot_and_save(error, "Difference Map", "difference.png")

            if config.ERROR_CALC:
                mse = la.norm(error.ravel(), np.inf)
            else:
                mse = la.norm(error)
            with open("results.pkl", "wb") as f:
                results = {
                    "original_values": true_values_on_grid,
                    "approximation": approximated_values_on_grid,
                    "errors": error,
                    "mse": mse,
                    "mesh_norm": fill_distance,
                }
                pkl.dump(results, f)

        yield mse, fill_distance, error


def run_all_experiments(diffs):
    mses = dict()
    fill_distances = dict()
    calculation_time = list()
    mus = list()
    path = "{}_{}".format(config.EXECUTION_NAME, time.strftime("%Y%m%d__%H%M%S"))

    with set_output_directory(path):
        for diff in diffs:
            config.renew()
            config.update_config_with_diff(diff)

            t_0 = datetime.now()

            for mse, fill_distance, _ in run_single_experiment():
                # log results
                mus.append(config.SCALING_FACTOR)
                t_f = datetime.now()
                calculation_time.append(t_f - t_0)
                mse_label = config.MSE_LABEL
                current_mses = mses.get(mse_label, list())
                current_mesh_norms = fill_distances.get(mse_label, list())
                current_mses.append(np.log(mse))
                current_mesh_norms.append(np.log(fill_distance))
                mses[mse_label] = current_mses
                fill_distances[mse_label] = current_mesh_norms
                t_0 = datetime.now()

        plot_lines(
            fill_distances,
            mses,
            "mses.svg",
            "Errors Comparison",
            "log$(h_X)$",
            "log(Error)",
        )
        result = {"mses": mses, "mesh_norms": fill_distances, "mus": mus}
        with open("results_dict.pkl", "wb") as f:
            pkl.dump(result, f)

    print("MSEs are: {}".format(mses))
    print("mesh_norms are: {}".format(fill_distances))
    print("times are: {}".format(calculation_time))
    return mses
