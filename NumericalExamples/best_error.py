"""
Finding configuration with best error.
"""

import time
from matplotlib import pyplot as plt

import Experiment
from Config.Config import config
from Config.Options import options
from Tools.Results import ResultsStorage
from Tools.Utils import set_output_directory, plot_lines

NUMBER_OF_SCALES = 7
SCALING_FACTORS = [0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 0.95, 1]
LAST_SCALE = min(SCALING_FACTORS) ** NUMBER_OF_SCALES


def build_diffs(number_of_scales):
    base_config = {
        "MANIFOLD": options.get_option("manifold", "numbers")(),
        "ORIGINAL_FUNCTION": options.get_option("original_function", "numbers_sin"),
        "EXECUTION_NAME": "multi_scale_with_same_end_scale",
        "SCALED_INTERPOLATION_METHOD": "quasi",
        "NUMBER_OF_SCALES": number_of_scales,
        "TEST_FILL_DISTANCE": 0.015,
    }

    config.set_base_config(base_config)
    config.renew()
    diffs = list()

    base_scales = [LAST_SCALE / scale ** number_of_scales for scale in SCALING_FACTORS]

    for scaling_factor, base_scale in zip(SCALING_FACTORS, base_scales):
        diff = {
            "NAME": f"scaling_factor_{scaling_factor}",
            "BASE_SCALE": base_scale,
            "MSE_LABEL": f"Multiscale with scaling factor {scaling_factor}",
            "SCALING_FACTOR": scaling_factor,
        }
        diffs.append(diff)

    return diffs


def run_experiment(diffs):
    results = Experiment.run_all_experiments(diffs)
    results["mus"] = results["mus"][::NUMBER_OF_SCALES]

    return results


def main():
    numbers_of_scales = range(1, NUMBER_OF_SCALES + 1)
    diffs_for_scale = [build_diffs(n) for n in numbers_of_scales]
    error_rates = ResultsStorage()
    times = ResultsStorage()

    for i, diffs in enumerate(diffs_for_scale):
        results = run_experiment(diffs)
        label_name = f"$N_f=${i+1}"

        # fit_multi_and_single(results)
        with set_output_directory(results["path"]):
            plt.figure()
            plt.plot(SCALING_FACTORS, [v[-1] for v in results["mses"].values()])
            plt.savefig("best_error.png")

            plt.figure()
            plt.plot(SCALING_FACTORS, [sum(v) for v in results["times"].values()])
            plt.savefig("times.png")

        for error, calculation_time in zip(
            results["mses"].values(), results["times"].values()
        ):
            error_rates.append(error[-1], label_name)
            times.append(sum(calculation_time), label_name)

    plot_lines(
        SCALING_FACTORS,
        error_rates.results,
        "error_rates.png",
        0,
        "$\mu$ - scaling factor",
        "error on final scale",
    )

    plot_lines(
        SCALING_FACTORS,
        times.results,
        "times.png",
        0,
        "$\mu$ - scaling factor",
        "Total calculation time",
    )


if __name__ == "__main__":
    with set_output_directory("results"):
        path = "{}_{}".format("optimization", time.strftime("%Y%m%d__%H%M%S"))
        with set_output_directory(path):
            main()
