import pickle as pkl
import time
from datetime import datetime

from Config import CONFIG, DIFFS
from Tools.SamplingPoints import Grid, symmetric_grid_params
from Tools.Utils import *


def multiscale_interpolation(manifold,
                             original_function,
                             grid_size,
                             resolution,
                             scaling_factor,
                             rbf,
                             number_of_scales,
                             scaled_interpolation_method,
                             is_approximating_on_tangent,
                             is_adaptive,
                             kernel_normalizer):
    f_j = manifold.zero_func
    e_j = act_on_functions(manifold.log, f_j, original_function)
    # for scale, kernel_normalizer in [(0.75 ** 4, 1/108.7)]:
    for scale, kernel_normalizer in [(0.75, 1 / 16.6), (0.75 ** 2, 1 / 30.4), (0.75 ** 3, 1 / 58.4),
                                     (0.75 ** 4, 1 / 108.7)]:
        # for scale_index in range(1, number_of_scales + 1):
        #     scale = scaling_factor ** scale_index
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
            scale / resolution,
            is_approximating_on_tangent,
            kernel_normalizer
        )
        s_j = method.approximation
        print("interpolated!")

        if is_approximating_on_tangent or is_adaptive:
            function_added_to_f_j = s_j
        else:
            function_added_to_f_j = act_on_functions(manifold.log, manifold.zero_func, s_j)

        f_j = act_on_functions(manifold.exp, f_j, function_added_to_f_j)
        e_j = act_on_functions(manifold.log, f_j, original_function)
        yield f_j, method


def run_single_experiment(config, rbf_generator, original_function):
    grid_size = config["GRID_SIZE"]
    base_resolution = config["BASE_RESOLUTION"]
    number_of_scales = config["NUMBER_OF_SCALES"]
    test_mesh_norm = config["TEST_MESH_NORM"]
    scaling_factor = config["SCALING_FACTOR"]
    experiment_name = config["NAME"] or "temp"
    manifold = config["MANIFOLD"]
    scaled_interpolation_method = config["SCALED_INTERPOLATION_METHOD"]
    norm_visualization = config["NORM_VISUALIZATION"]
    is_approximating_on_tangent = config["IS_APPROXIMATING_ON_TANGENT"]
    is_adaptive = config["IS_ADAPTIVE"]
    kernel_normalizer = config["KERNEL_NORMALIZER"]

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
                  "derivatives.png")

    for i, (interpolant, method) in enumerate(multiscale_interpolation(
            manifold,
            number_of_scales=number_of_scales,
            original_function=original_function,
            grid_size=grid_size,
            resolution=base_resolution,
            scaling_factor=scaling_factor,
            rbf=rbf_generator(base_resolution),
            scaled_interpolation_method=scaled_interpolation_method,
            is_approximating_on_tangent=is_approximating_on_tangent,
            is_adaptive=is_adaptive,
            kernel_normalizer=kernel_normalizer
    )):
        with set_output_directory("{}_{}".format(experiment_name, i + 1)):
            with open("config.pkl", "wb") as f:
                pkl.dump(config, f)

            approximated_values_on_grid = Grid(1, interpolant, grid_params).evaluation
            print(f"Histogram: {method.grid_histogram}")

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
    x_data_sets = dict()
    calculation_time = list()
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
                # mse = np.log(mse)
                mse_label = current_config["MSE_LABEL"]
                current_mses = mses.get(mse_label, list())
                current_x_data = x_data_sets.get(mse_label, list())
                current_mses.append(mse)
                # current_x_data.append(np.log(current_config["SCALING_FACTOR"]))
                current_x_data.append((current_config["SCALING_FACTOR"]))
                mses[mse_label] = current_mses
                x_data_sets[mse_label] = current_x_data
                t_0 = datetime.now()

        plot_lines(mses, "mses.png", "Error in different runs", "Scaling factor", "Error", x_data=x_data_sets)

    print("MSEs are: {}".format(mses))
    print("times are: {}".format(calculation_time))
    return mses


def main():
    rbf_generator = generate_wendland
    original_function = CONFIG["ORIGINAL_FUNCTION"]
    config = CONFIG
    output_dir = CONFIG["OUTPUT_DIR"]
    diffs = DIFFS

    with set_output_directory(output_dir):
        results = run_all_experiments(config, diffs, rbf_generator, original_function)

    return results, interpolants


if __name__ == "__main__":
    _, interpolants = main()
