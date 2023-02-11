import Experiment
from Config.Config import config
from Config.Options import options
from Tools.Utils import set_output_directory

NUMBER_OF_SCALES = 4


def run_multiscale_vs_single_scale(function):
    base_config = {
        "MANIFOLD": options.get_option("manifold", "rotations")(),
        "NUMBER_OF_SCALES": NUMBER_OF_SCALES,
        "SCALING_FACTOR": 0.75,
        "TEST_FILL_DISTANCE": 0.02,
        "ORIGINAL_FUNCTION": options.get_option("original_function", function),
        "EXECUTION_NAME": f"quasi_interpolation_vs_multiscale_{function}",
        "SCALED_INTERPOLATION_METHOD": "quasi",
        "DATA_SITES_GENERATION": "halton",
        "DATA_SITES_STORAGE": "kd-tree",
        "IS_APPROXIMATING_ON_TANGENT": False,
        "NOISE": "rotation_gaussian_noise",
        "DENOISE": True,
        "DENOISE_THRESHOLD": 0.9,
        "NOISE_SIGMA": 0.2,
        "CMAX": 1.3,
    }

    config.set_base_config(base_config)
    config.renew()

    diffs = list()

    # Add multiscale iterations
    diffs.append(
        {
            "MSE_LABEL": "Multiscale",
            "NAME": "Multiscale",
        }
    )

    with set_output_directory("results"):
        Experiment.run_all_experiments(diffs)


def main():
    original_functions = ["rotations_euler"]

    for function in original_functions:
        run_multiscale_vs_single_scale(function)


if __name__ == "__main__":
    main()
