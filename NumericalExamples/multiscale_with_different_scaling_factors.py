import Experiment
from Config.Config import config
from Config.Options import options
from Tools.Utils import set_output_directory

NUMBER_OF_SCALES = 4


def main():
    base_config = {
        "MANIFOLD": options.get_option("manifold", "numbers")(),
        "ORIGINAL_FUNCTION": options.get_option("original_function", "numbers_sin"),
        "NUMBER_OF_SCALES": NUMBER_OF_SCALES,
        "SCALING_FACTOR": 0.75,
        "EXECUTION_NAME": "multiscale_with_different_scaling_factors",
        "SCALED_INTERPOLATION_METHOD": "quasi",
    }

    config.set_base_config(base_config)
    config.renew()

    scaling_factors = [0.5, 0.6, 0.65, 0.7, 0.75]
    diffs = list()

    for scaling_factor in scaling_factors:
        diff = {
            "NAME": f"scaling_factor_{scaling_factor}",
            "MSE_LABEL": f"Multiscale with scaling factor {scaling_factor}",
            "SCALING_FACTOR": scaling_factor,
        }
        diffs.append(diff)

    with set_output_directory("results"):
        Experiment.run_all_experiments(diffs)


if __name__ == "__main__":
    main()
