import argparse
import importlib
from itertools import product

import numpy as np

from Config.Options import options
from Config.Config import config
import Experiment
from Tools.Utils import set_output_directory
from DataSites.PolynomialReproduction import condition_g


def parse_arguments():
    parser = argparse.ArgumentParser("RBF Approximation Script")
    parser.add_argument(
        "-f",
        "--function",
        type=str,
        help="Original function for approximation, see OriginalFunction.py",
        choices=options.get_options("original_function"),
        default="numbers",
    )
    parser.add_argument(
        "-m",
        "--manifold",
        choices=options.get_options("manifold").keys(),
        default="numbers",
    )
    parser.add_argument(
        "-t",
        "--tangent-approximation",
        action="store_true",
        help="Should approximate using tangent averaging?",
    )
    parser.add_argument(
        "-nv",
        "--norm-visualization",
        action="store_true",
        help="Should visualize quickly using norm visualization",
    )
    parser.add_argument(
        "-s",
        "--single-scale",
        action="store_true",
        help="Should approximate the single scale case?",
    )
    parser.add_argument("-n", "--number-of-scales", type=int, default=1)
    parser.add_argument(
        "-b", "--base-index", type=int, help="The first number of scales", default=1
    )
    parser.add_argument(
        "-sf", "--scaling-factor", type=float, default=config.SCALING_FACTOR
    )
    parser.add_argument("-e", "--execution-name", type=str, default="NoName")
    parser.add_argument("-a", "--adaptive", action="store_true", help="is adaptive m0")
    parser.add_argument(
        "-mt",
        "--method",
        choices=options.get_options("approximation_method").keys(),
        default="quasi",
        help="approximation method",
    )
    parser.add_argument("-ci", "--compare-to-interpolation", action="store_true")
    parser.add_argument("-dm", "--dont-multi", action="store_true")
    parser.add_argument(
        "-cal", "--calibrate", action="store_true", help="config through nonorm cache"
    )
    parser.add_argument("-er", "--error", action="store_true")
    parser.add_argument("-mu", "--mu-testing", action="store_true")
    parser.add_argument("-df", "--different-functions", action="store_true")
    args = parser.parse_args()

    base_config = dict()
    base_config["ORIGINAL_FUNCTION"] = options.get_option(
        "original_function", args.function
    )
    base_config["ERROR_CALC"] = args.error
    base_config["MANIFOLD"] = options.get_option("manifold", args.manifold)()
    base_config["IS_APPROXIMATING_ON_TANGENT"] = args.tangent_approximation
    base_config["NORM_VISUALIZATION"] = args.norm_visualization
    base_config["SCALING_FACTOR"] = args.scaling_factor
    is_tangent = "Tangent" if args.tangent_approximation else "Intrinsic"
    base_config["EXECUTION_NAME"] = "{}_{}".format(args.manifold, is_tangent)
    execution_name = (
        args.execution_name
        if (args.execution_name != "NoName")
        else "{}_{}".format(args.manifold, is_tangent)
    )
    base_config["EXECUTION_NAME"] = execution_name
    base_config["IS_ADAPTIVE"] = args.adaptive
    base_config["SCALED_INTERPOLATION_METHOD"] = args.method
    base_config["CALIBRATE"] = args.calibrate

    config.set_base_config(base_config)
    config.renew()

    if args.different_functions:
        return run_different_functions(args)

    if args.mu_testing:
        return run_different_mus(args)

    if args.dont_multi:
        diffs = []
    else:
        diffs = [
            {
                "NAME": "multiscale",
                "NUMBER_OF_SCALES": args.base_index + args.number_of_scales - 1,
                "MSE_LABEL": "Multiscale",
            }
        ]

    if args.single_scale:
        diffs = diffs + [
            {
                "NAME": "single_scale_{}".format(index),
                "MSE_LABEL": "Single scale",
                "NUMBER_OF_SCALES": 1,
                "SCALING_FACTOR": args.scaling_factor ** index,
                "SCALING_FACTOR_POWER": index,
            }
            for index in range(args.base_index, args.base_index + args.number_of_scales)
        ]

    if args.compare_to_interpolation:
        print("Comparing")
        diffs_copy = diffs.copy()
        for item in diffs_copy:
            new_item = item.copy()
            new_item["NAME"] = "_".join([item["NAME"], "interpolation"])
            new_item["MSE_LABEL"] = " ".join(["Interpolated", item["MSE_LABEL"]])
            new_item["SCALED_INTERPOLATION_METHOD"] = "naive"
            diffs.append(new_item)

    return diffs


def run_different_mus(args):
    mus = [0.5, 0.6, 0.65, 0.7, 0.75]
    diffs = list()

    for mu in mus:
        diff = {
            "NAME": f"different_mus_{mu}",
            "MSE_LABEL": f"Multiscale with scaling factor {mu}",
            "NUMBER_OF_SCALES": args.base_index + args.number_of_scales - 1,
            "SCALING_FACTOR": mu,
        }
        diffs.append(diff)
    return diffs


def run_different_functions(args):
    functions = [
        "numbers_gauss",
        "numbers_high_freq",
        "numbers_low_freq",
        "numbers_sin",
        "numbers_gauss_freqs",
    ]
    diffs = []

    for function_name in functions:
        diff = {
            "ORIGINAL_FUNCTION": options.get_option("original_function", function_name),
            "NAME": f"multiscale_{function_name}",
            "MSE_LABEL": f"multiscale_{function_name}",
            "NUMBER_OF_SCALES": args.number_of_scales,
        }
        diffs.append(diff)

        diffs = diffs + [
            {
                "NAME": "single_scale_{}_{}".format(function_name, index),
                "ORIGINAL_FUNCTION": options.get_option("original_fucntion", function_name),
                "MSE_LABEL": f"Single_Scale_{function_name}",
                "NUMBER_OF_SCALES": 1,
                "SCALING_FACTOR": args.scaling_factor ** index,
                "SCALING_FACTOR_POWER": index,
            }
            for index in range(args.base_index, args.base_index + args.number_of_scales)
        ]

    return diffs


def run_different_rbfs(diffs):
    old_diffs = diffs.copy()
    diffs = list()

    wendlands = [
        "wendland_3_1"
    ]

    methods = [
        "moving",
        "quasi",
    ]

    mus = [
        # 0.5,
        # 0.6,
        # 0.65,
        # 0.7,
        0.75,
        # 0.8,
    ]

    functions = [
        # "numbers_gauss",
        # "numbers_high_freq",
        # "numbers_low_freq",
        "numbers_sin",
        # "numbers_gauss_freqs",
    ]

    for wendland_name, method_name, mu, function in product(
        wendlands, methods, mus, functions
    ):
        wendland = wendlands[wendland_name]
        method = methods[method_name]
        for diff in old_diffs:
            current_diff = diff.copy()
            current_diff[
                "NAME"
            ] = f"{method_name}__{wendland_name}__mu_{mu}__{function}__{current_diff['NAME']}"
            current_diff[
                "MSE_LABEL"
            ] = f"{method_name}__{wendland_name}__mu_{mu}__{function}__{current_diff['MSE_LABEL']}"
            current_diff["RBF"] = wendland
            current_diff["SCALED_INTERPOLATION_METHOD"] = method
            current_diff["SCALING_FACTOR"] = mu ** current_diff.get(
                "SCALING_FACTOR_POWER", config.SCALING_FACTOR_POWER
            )
            current_diff["ORIGINAL_FUNCTION"] = options.get_option("original_function", function)
            diffs.append(current_diff)

    return diffs


def run_quasi_comparison(diffs):
    old_diffs = diffs.copy()
    diffs = list()

    wendland = "wendland_3_1"
    methods = {
        "Quadratic Reproduction": "moving",
        "Constant Reproduction": "quasi",
    }
    mu = 0.75

    function = "numbers_sin"

    for method_name in methods:
        method = methods[method_name]
        for diff in old_diffs:
            current_diff = diff.copy()
            current_diff[
                "NAME"
            ] = f"{method_name}__{wendland}__mu_{mu}__{function}__{current_diff['NAME']}"
            current_diff["MSE_LABEL"] = f"{method_name}"
            current_diff["RBF"] = wendland
            current_diff["SCALED_INTERPOLATION_METHOD"] = method
            current_diff["SCALING_FACTOR"] = mu ** current_diff.get(
                "SCALING_FACTOR_POWER", config.SCALING_FACTOR_POWER
            )
            current_diff["ORIGINAL_FUNCTION"] = options.get_option("original_function", function)
            diffs.append(current_diff)

    return diffs


def run_functions_comparison(diffs):
    old_diffs = diffs.copy()

    wendland = "wendland_3_1"
    method = "quasi"
    mu = 0.75
    execution_name = config.EXECUTION_NAME

    functions = ["numbers", "numbers_gauss"]
    for function in functions:
        diffs = list()
        config.EXECUTION_NAME = f"{execution_name}_{function}"
        for diff in old_diffs:
            current_diff = diff.copy()
            current_diff["NAME"] = f"{function}__{current_diff['NAME']}"
            current_diff["RBF"] = wendland
            current_diff["SCALED_INTERPOLATION_METHOD"] = method
            current_diff["SCALING_FACTOR"] = mu ** current_diff.get(
                "SCALING_FACTOR_POWER", config.SCALING_FACTOR_POWER
            )
            current_diff["ORIGINAL_FUNCTION"] = options.get_option("original_function", function)
            diffs.append(current_diff)

        Experiment.run_all_experiments(diffs)

    return config, diffs


def main():
    diffs = parse_arguments()
    original_function = config.ORIGINAL_FUNCTION
    output_dir = config.OUTPUT_DIR

    # diffs = run_different_rbfs(diffs)
    diffs = run_quasi_comparison(diffs)

    with set_output_directory(output_dir):
        # return run_functions_comparison(diffs)

        if not config.CALIBRATE:
            results = Experiment.run_all_experiments(diffs)
        else:
            results = Experiment.calibrate(diffs)

    if len(condition_g):
        print(f"Average condition {np.average(condition_g)}")


if __name__ == "__main__":
    main()
