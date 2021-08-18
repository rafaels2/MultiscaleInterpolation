"""
Flexible run of the multiscale approximation experiments
"""
import argparse
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
    parser.add_argument("-dm", "--dont-multi", action="store_true")
    args = parser.parse_args()

    base_config = dict()
    base_config["ORIGINAL_FUNCTION"] = options.get_option(
        "original_function", args.function
    )
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

    config.set_base_config(base_config)
    config.renew()

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

    return diffs


def main():
    diffs = parse_arguments()
    output_dir = config.OUTPUT_DIR

    with set_output_directory(output_dir):
        results = Experiment.run_all_experiments(diffs)

    if len(condition_g):
        print(f"Average condition {np.average(condition_g)}")


if __name__ == "__main__":
    main()
