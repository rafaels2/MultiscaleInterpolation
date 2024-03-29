from datetime import datetime
import pickle as pkl
import numpy as np
import time

from Config.Config import config
from Config.Options import options
from DataSites.Generation.Grid import get_grid
from DataSites.GridUtils import calculate_max_derivative
from DataSites.Storage.Grid import Grid
from Tools.Results import ResultsStorage
from Tools.Utils import *
from DataSites.GridUtils import symmetric_grid_params

# Configure plot style
config_plt(plt)


def multiscale_approximation():
    """
    Run multiscale approximation
    """

    # approximate when initial guess f_0 = 0
    f_j = config.MANIFOLD.zero_func

    # Initial error e_0 = log(0, f_j)
    e_j = act_on_functions(config.MANIFOLD.log, f_j, config.ORIGINAL_FUNCTION)

    # For all scales do
    for scale_index in range(1, config.NUMBER_OF_SCALES + 1):
        scale = config.BASE_SCALE * config.SCALING_FACTOR ** scale_index

        if config.IS_APPROXIMATING_ON_TANGENT:
            function_to_interpolate = e_j
        elif config.IS_ADAPTIVE:
            function_to_interpolate = (
                e_j,
                act_on_functions(config.MANIFOLD.exp, config.MANIFOLD.zero_func, e_j),
            )
        else:
            # Approximate exp(0, e_j)
            function_to_interpolate = act_on_functions(
                config.MANIFOLD.exp, config.MANIFOLD.zero_func, e_j
            )

        # Initializing current scale sites properties
        fill_distance = scale / config.BASE_RESOLUTION
        current_grid_parameters = symmetric_grid_params(
            config.GRID_SIZE + config.GRID_BORDER, fill_distance
        )

        # Call the approximation method
        approximation_method = options.get_option(
            "approximation_method", config.SCALED_INTERPOLATION_METHOD
        )(
            function_to_interpolate,
            current_grid_parameters,
            scale,
        )

        # s_j = Q(e_j)
        s_j = approximation_method.approximation

        if config.IS_APPROXIMATING_ON_TANGENT or config.IS_ADAPTIVE:
            function_added_to_f_j = s_j
        else:
            function_added_to_f_j = act_on_functions(
                config.MANIFOLD.log, config.MANIFOLD.zero_func, s_j
            )

        # f_j = exp (f_{j-1}, s_j)
        f_j = act_on_functions(config.MANIFOLD.exp, f_j, function_added_to_f_j)

        # Update the error for next step
        e_j = act_on_functions(config.MANIFOLD.log, f_j, config.ORIGINAL_FUNCTION)
        yield fill_distance, f_j


def calculate_execution_time(func):
    def new_func():
        t_0 = datetime.now()
        for ans in func():
            ans = list(ans)
            t_f = datetime.now()
            ans.insert(0, (t_f - t_0).total_seconds())
            yield tuple(ans)
            t_0 = datetime.now()

    return new_func


@calculate_execution_time
def run_single_experiment():
    """ Run an experiment with the current config """

    # Initialize test grid
    grid_params = symmetric_grid_params(config.GRID_SIZE, config.TEST_FILL_DISTANCE)
    sites = get_grid(*grid_params)

    # Evaluate original function on the grid
    true_values_on_grid = Grid(
        sites, 1, config.ORIGINAL_FUNCTION, grid_params.fill_distance
    ).evaluation

    # Plot the original evaluation
    config.MANIFOLD.plot(
        true_values_on_grid,
        "Original",
        "original.png",
        norm_visualization=config.NORM_VISUALIZATION,
    )

    # Plot max derivatives
    plot_and_save(
        calculate_max_derivative(
            config.ORIGINAL_FUNCTION, grid_params, config.MANIFOLD
        ),
        "Max Derivatives",
        "derivatives.png",
    )

    # Run multiscale iterations
    for i, (fill_distance, interpolant) in enumerate(multiscale_approximation()):
        # Each scale in the multiscale, evaluate and save the error
        with set_output_directory("{}_{}".format(config.NAME, i + 1)):
            # Save the results of current scale
            with open("config.pkl", "wb") as f:
                # pkl.dump(config, f)
                pass

            # Evaluate the approximation on the test grid
            sites = get_grid(*grid_params)
            approximated_values_on_grid = Grid(
                sites, 1, interpolant, grid_params.fill_distance
            ).evaluation

            # Plot the evaluation
            config.MANIFOLD.plot(
                approximated_values_on_grid,
                "Approximation",
                "approximation.png",
                norm_visualization=config.NORM_VISUALIZATION,
            )

            # Calculate and plot the current scale's approximation error.
            error = config.MANIFOLD.calculate_error(
                approximated_values_on_grid, true_values_on_grid
            )
            plot_and_save(error, "Difference Map", "difference.png")

            # Calculate the l_2 norm of the error
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
    """ Experiments runner, gets a list of config differences for each iteration """
    mses = ResultsStorage()
    fill_distances = ResultsStorage()
    calculation_times = ResultsStorage()
    mus = list()

    # Output of the run is in results/path
    path = "{}_{}".format(config.EXECUTION_NAME, time.strftime("%Y%m%d__%H%M%S"))

    with set_output_directory(path):
        for diff in diffs:
            # Update configurations
            config.renew()
            config.update_config_with_diff(diff)

            # Run the iteration
            for calculation_time, mse, fill_distance, _ in run_single_experiment():
                # log results
                mse_label = config.MSE_LABEL
                calculation_times.append(calculation_time, mse_label)
                mses.append(np.log(mse), mse_label)
                fill_distances.append(np.log(fill_distance), mse_label)
                mus.append(config.SCALING_FACTOR)

        # Plot error rates comparison
        plot_lines(
            fill_distances.results,
            mses.results,
            "mses.svg",
            "Errors Comparison",
            "log$(h_X)$",
            "log(Error)",
        )

        result = {
            "mses": mses.results,
            "mesh_norms": fill_distances.results,
            "mus": mus,
            "times": calculation_times.results,
            "path": path,
        }
        with open("results_dict.pkl", "wb") as f:
            pkl.dump(result, f)

        plot_lines(
            fill_distances.results,
            calculation_times.results,
            "time_comparison.png",
            "Time Comparison",
            "$log(h_X)$",
            "time",
        )

    print("MSEs are: {}".format(mses))
    print("mesh_norms are: {}".format(fill_distances))
    print("times are: {}".format(calculation_times))
    return result
