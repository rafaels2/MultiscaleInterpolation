import Experiment
from Config.Config import config
from Config.Options import options
from Tools.Utils import set_output_directory

NUMBER_OF_SCALES = 5


def run_multiscale_vs_single_scale(function):
    base_config = {
        "MANIFOLD": options.get_option("manifold", "numbers")(),
        "NUMBER_OF_SCALES": NUMBER_OF_SCALES,
        "SCALING_FACTOR": 0.8,
        "ORIGINAL_FUNCTION": options.get_option("original_function", function),
        "EXECUTION_NAME": f"quasi_interpolation_vs_multiscale_{function}",
        "SCALED_INTERPOLATION_METHOD": "quasi",
        "DATA_SITES_GENERATION": "halton",
        "DATA_SITES_STORAGE": "kd-tree",
    }

    config.set_base_config(base_config)
    config.renew()

    diffs = list()

    # Add multiscale iterations
    diffs.append(
        {
            "MSE_LABEL": "Multiscale",
            "NAME": "Multiscale",
            "NUMBER_OF_SCALES": NUMBER_OF_SCALES,
        }
    )

    # Add single scale iterations
    for iteration in range(1, NUMBER_OF_SCALES + 1):
        current_diff = {
            "MSE_LABEL": "Single Scale",
            "NUMBER_OF_SCALES": 1,
            "NAME": f"Single_Scale_{iteration}",
            "SCALING_FACTOR": config.SCALING_FACTOR ** iteration,
            "SCALING_FACTOR_POWER": iteration,
        }
        diffs.append(current_diff)

    with set_output_directory("results"):
        Experiment.run_all_experiments(diffs)


def main():
    original_functions = ["numbers_non_smooth", "numbers", "numbers_gauss"]

    for function in original_functions:
        run_multiscale_vs_single_scale(function)


if __name__ == "__main__":
    main()
