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
        "EXECUTION_NAME": f"different_denoising_thresholds",
        "SCALED_INTERPOLATION_METHOD": "quasi",
        "DATA_SITES_GENERATION": "halton",
        "DATA_SITES_STORAGE": "kd-tree",
        "IS_APPROXIMATING_ON_TANGENT": True,
        "NOISE": "rotation_gaussian_noise",
        "DENOISE": True,
        "NOISE_SIGMA": 0.1,
    }

    config.set_base_config(base_config)
    config.renew()

    diffs = list()

    # Add multiscale iterations
    for threshold in range(75, 100, 5):
        threshold_formatted = f"0.{threshold}"
        diffs.append(
            {
                "MSE_LABEL": f"t = {threshold_formatted}",
                "NAME": f"threshold_{threshold}",
                "DENOISE_THRESHOLD": 0.01 * threshold,
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
