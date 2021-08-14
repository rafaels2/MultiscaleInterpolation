import Experiment
from Config.Config import config
from Config.Options import options


def main():
    base_config = {
        "MANIFOLD": options.get_option("manifold", "numbers")(),
        "ORIGINAL_FUNCTION": options.get_option("original_function", "numbers_sin"),
        "NUMBER_OF_SCALES": 5,
        "SCALING_FACTOR": 0.75,
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
        current_diff["NAME"] = f"{method_name}"
        current_diff["MSE_LABEL"] = f"{method_name}"
        current_diff["SCALED_INTERPOLATION_METHOD"] = method
        diffs.append(current_diff)

    Experiment.run_all_experiments(diffs)


if __name__ == "__main__":
    main()
