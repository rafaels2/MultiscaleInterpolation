import argparse
import importlib
from itertools import product

import numpy as np

import Experiment
from ApproximationMethods.AdaptiveQuasi import AdaptiveQuasi
from ApproximationMethods.MovingLeastSquares import MovingLeastSquares
from ApproximationMethods.Naive import Naive
from ApproximationMethods.NoNormalization import NoNormalization
from ApproximationMethods.Quasi import Quasi
from Config import CONFIG, _SCALING_FACTOR
from Tools.Utils import (
    set_output_directory,
    wendland_3_0,
    wendland_3_1,
    wendland_3_2,
    wendland_1_0,
)
from Manifolds import MANIFOLDS
import ExampleFunctions
from Tools.Lambdas import condition_g

METHODS = {
    "naive": Naive,
    "quasi": Quasi,
    "adaptive": AdaptiveQuasi,
    "moving": MovingLeastSquares,
}


def parse_arguments():
    parser = argparse.ArgumentParser("RBF Approximation Scrtipt")
    parser.add_argument(
        "-f", "--function", type=str, help="Original function file", required=True
    )
    parser.add_argument("-m", "--manifold", choices=MANIFOLDS.keys(), required=True)
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
    parser.add_argument("-sf", "--scaling-factor", type=float, default=_SCALING_FACTOR)
    parser.add_argument("-e", "--execution-name", type=str, default="NoName")
    parser.add_argument("-a", "--adaptive", action="store_true", help="is adaptive m0")
    parser.add_argument(
        "-mt",
        "--method",
        choices=METHODS.keys(),
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

    config = CONFIG.copy()
    config["ORIGINAL_FUNCTION"] = importlib.import_module(
        args.function
    ).original_function
    config["ERROR_CALC"] = args.error
    config["MANIFOLD"] = MANIFOLDS[args.manifold]()
    config["IS_APPROXIMATING_ON_TANGENT"] = args.tangent_approximation
    config["NORM_VISUALIZATION"] = args.norm_visualization
    config["SCALING_FACTOR"] = args.scaling_factor
    is_tangent = "Tangent" if args.tangent_approximation else "Intrinsic"
    config["EXECUTION_NAME"] = "{}_{}".format(args.manifold, is_tangent)
    execution_name = (
        args.execution_name
        if (args.execution_name != "NoName")
        else "{}_{}".format(args.manifold, is_tangent)
    )
    config["EXECUTION_NAME"] = execution_name
    config["IS_ADAPTIVE"] = args.adaptive
    config["SCALED_INTERPOLATION_METHOD"] = METHODS[args.method]
    config["CALIBRATE"] = args.calibrate

    if args.different_functions:
        return config, run_different_functions(args)

    if args.mu_testing:
        return config, run_different_mus(config, args)

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
            new_item["SCALED_INTERPOLATION_METHOD"] = Naive
            diffs.append(new_item)

    return config, diffs


def run_different_mus(config, args):
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
            "ORIGINAL_FUNCTION": importlib.import_module(
                f"ExampleFunctions.{function_name}"
            ).original_function,
            "NAME": f"multiscale_{function_name}",
            "MSE_LABEL": f"multiscale_{function_name}",
            "NUMBER_OF_SCALES": args.number_of_scales,
        }
        diffs.append(diff)

        diffs = diffs + [
            {
                "NAME": "single_scale_{}_{}".format(function_name, index),
                "ORIGINAL_FUNCTION": importlib.import_module(
                    f"ExampleFunctions.{function_name}"
                ).original_function,
                "MSE_LABEL": f"Single_Scale_{function_name}",
                "NUMBER_OF_SCALES": 1,
                "SCALING_FACTOR": args.scaling_factor ** index,
                "SCALING_FACTOR_POWER": index,
            }
            for index in range(args.base_index, args.base_index + args.number_of_scales)
        ]

    return diffs


def run_different_rbfs(config, diffs):
    old_diffs = diffs.copy()
    diffs = list()

    wendlands = {
        # "1_0": wendland_1_0,
        # "3_0": wendland_3_0,
        "3_1": wendland_3_1,
        # "3_2": wendland_3_2,
    }

    methods = {
        "moving": MovingLeastSquares,
        "quasi": Quasi,
        # "no_normalization": NoNormalization
    }

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
                "SCALING_FACTOR_POWER", config["SCALING_FACTOR_POWER"]
            )
            current_diff["ORIGINAL_FUNCTION"] = importlib.import_module(
                f"ExampleFunctions.{function}"
            ).original_function
            diffs.append(current_diff)

    return config, diffs


def run_quasi_comparison(config, diffs):
    old_diffs = diffs.copy()
    diffs = list()

    wendland = wendland_3_1
    methods = {
        "Quadratic Reproduction": MovingLeastSquares,
        "Constant Reproduction": Quasi,
    }
    mu = 0.75

    function = "numbers_sin"

    for method_name in methods:
        method = methods[method_name]
        for diff in old_diffs:
            current_diff = diff.copy()
            current_diff[
                "NAME"
            ] = f"{method_name}__{wendland.__name__}__mu_{mu}__{function}__{current_diff['NAME']}"
            current_diff["MSE_LABEL"] = f"{method_name}"
            current_diff["RBF"] = wendland
            current_diff["SCALED_INTERPOLATION_METHOD"] = method
            current_diff["SCALING_FACTOR"] = mu ** current_diff.get(
                "SCALING_FACTOR_POWER", config["SCALING_FACTOR_POWER"]
            )
            current_diff["ORIGINAL_FUNCTION"] = importlib.import_module(
                f"ExampleFunctions.{function}"
            ).original_function
            diffs.append(current_diff)

    return config, diffs


def run_functions_comparison(config, diffs):
    old_diffs = diffs.copy()

    wendland = wendland_3_1
    method = Quasi
    mu = 0.75
    execution_name = config["EXECUTION_NAME"]

    functions = ["numbers", "numbers_gauss"]
    for function in functions:
        diffs = list()
        config["EXECUTION_NAME"] = f"{execution_name}_{function}"
        for diff in old_diffs:
            current_diff = diff.copy()
            current_diff["NAME"] = f"{function}__{current_diff['NAME']}"
            current_diff["RBF"] = wendland
            current_diff["SCALED_INTERPOLATION_METHOD"] = method
            current_diff["SCALING_FACTOR"] = mu ** current_diff.get(
                "SCALING_FACTOR_POWER", config["SCALING_FACTOR_POWER"]
            )
            current_diff["ORIGINAL_FUNCTION"] = importlib.import_module(
                f"ExampleFunctions.{function}"
            ).original_function
            diffs.append(current_diff)

        Experiment.run_all_experiments(config, diffs, current_diff["ORIGINAL_FUNCTION"])

    return config, diffs


def main():
    config, diffs = parse_arguments()
    original_function = config["ORIGINAL_FUNCTION"]
    output_dir = CONFIG["OUTPUT_DIR"]

    # config, diffs = run_different_rbfs(config, diffs)
    config, diffs = run_quasi_comparison(config, diffs)

    with set_output_directory(output_dir):
        # return run_functions_comparison(config, diffs)

        if not config["CALIBRATE"]:
            results = Experiment.run_all_experiments(config, diffs, original_function)
        else:
            results = Experiment.calibrate(config, diffs, original_function)

    if len(condition_g):
        print(f"Average condition {np.average(condition_g)}")


if __name__ == "__main__":
    main()
