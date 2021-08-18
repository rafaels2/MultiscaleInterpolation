import Experiment
from Config.Config import config
from Config.Options import options
from Tools.Utils import set_output_directory

NUMBER_OF_SCALES = 5


def main():
    base_config = {
        "MANIFOLD": options.get_option("manifold", "numbers")(),
        "ORIGINAL_FUNCTION": options.get_option("original_function", "numbers_sin"),
        "NUMBER_OF_SCALES": 1,
        "SCALING_FACTOR": 0.75,
        "EXECUTION_NAME": "comparison_of_polynomial_reproduction_degree",
    }

    config.set_base_config(base_config)
    config.renew()

    methods = {
        "Quadratic Reproduction": "moving",
        "Constant Reproduction": "quasi",
    }

    diffs = list()

    for method_name in methods:
        current_diff = dict()
        method = methods[method_name]
        current_diff["MSE_LABEL"] = f"{method_name}"
        current_diff["SCALED_INTERPOLATION_METHOD"] = method
        for iteration in range(1, NUMBER_OF_SCALES + 1):
            current_diff["NAME"] = f"{method_name}_{iteration}"
            current_diff["SCALING_FACTOR_POWER"] = iteration
            current_diff["SCALING_FACTOR"] = config.SCALING_FACTOR ** iteration
            diffs.append(current_diff.copy())

    with set_output_directory("results"):
        Experiment.run_all_experiments(diffs)


if __name__ == "__main__":
    main()
