import argparse
import importlib

import Interpolation
from Config import CONFIG, _SCALING_FACTOR
from Tools.Utils import set_output_directory, generate_wendland
from Manifolds import MANIFOLDS
from ExampleFunctions.numbers_gauss import original_function


def parse_arguments():
    parser = argparse.ArgumentParser("RBF Approximation Script")
    parser.add_argument('-f', '--function', type=str, help='Original function file', required=True)
    parser.add_argument('-m', '--manifold', choices=MANIFOLDS.keys(), required=True)
    parser.add_argument('-t', '--tangent-approximation', action='store_true',
                        help='Should approximate using tangent averaging?')
    parser.add_argument('-nv', '--norm-visualization', action='store_true',
                        help='Should visualize quickly using norm visualization')
    parser.add_argument('-s', '--single-scale', action='store_true',
                        help='Should approximate the single scale case?')
    parser.add_argument('-n', '--number-of-scales', type=int, default=1)
    parser.add_argument('-b', '--base-index', type=int, help='The first number of scales', default=1)
    parser.add_argument('-sf', '--scaling-factor', type=float, default=_SCALING_FACTOR)
    parser.add_argument('-e', '--execution-name', type=str, default='NoName')
    parser.add_argument('-a', '--adaptive', action='store_true', help='is adaptive m0')
    args = parser.parse_args()

    config = CONFIG.copy()
    config['ORIGINAL_FUNCTION'] = importlib.import_module(args.function).original_function
    config['MANIFOLD'] = MANIFOLDS[args.manifold]()
    config['IS_APPROXIMATING_ON_TANGENT'] = args.tangent_approximation
    config['NORM_VISUALIZATION'] = args.norm_visualization
    config['SCALING_FACTOR'] = args.scaling_factor
    is_tangent = "Tangent" if args.tangent_approximation else "Intrinsic"
    config['EXECUTION_NAME'] = "{}_{}".format(args.manifold, is_tangent)
    execution_name = args.execution_name if (args.execution_name != 'NoName') else "{}_{}".format(args.manifold,
                                                                                                  is_tangent)
    config['EXECUTION_NAME'] = execution_name
    config['IS_ADAPTIVE'] = args.adaptive

    diffs = [
        {
            "NAME": "multiscale",
            "NUMBER_OF_SCALES": args.base_index + args.number_of_scales - 1,
            "MSE_LABEL": "Multi Scale"
        }]

    if args.single_scale:
        diffs = diffs + [
            {
                "NAME": "single_scale_{}".format(index),
                "MSE_LABEL": "Single scale",
                "NUMBER_OF_SCALES": 1,
                "SCALING_FACTOR": args.scaling_factor ** index
            } for index in range(args.base_index, args.base_index + args.number_of_scales)
        ]

    return config, diffs


def main(config=None, diffs=None):
    if config is None and diffs is None:
        config, diffs = parse_arguments()

    rbf_generator = generate_wendland
    current_original_function = config['ORIGINAL_FUNCTION']
    output_dir = CONFIG["OUTPUT_DIR"]

    with set_output_directory(output_dir):
        results = Interpolation.run_all_experiments(config, diffs, rbf_generator, current_original_function)

    return results


def generate_run_parameters(base_scaling_factor, n_sf, base_kernel_normalizer, n_kn):
    for i in range(n_sf):
        for j in range(n_kn):
            scaling_factor = base_scaling_factor ** (i + 1)
            kernel_normalizer = base_kernel_normalizer * (1 - 0.1 * j)
            yield scaling_factor, kernel_normalizer


def run_no_normalization_tests():
    config = CONFIG.copy()

    config['ORIGINAL_FUNCTION'] = original_function
    config['MANIFOLD'] = MANIFOLDS["numbers_no_normalization"]()
    config['IS_APPROXIMATING_ON_TANGENT'] = False
    config['NORM_VISUALIZATION'] = False
    config['SCALING_FACTOR'] = 1
    config['EXECUTION_NAME'] = "NoNormalization"
    config['IS_ADAPTIVE'] = False

    diffs = [
        {
            "NAME": f"sf_{scaling_factor}_kernel_normalizer_{kernel_normalizer}",
            "MSE_LABEL": f"kernel_normalizer_{kernel_normalizer}",
            "NUMBER_OF_SCALES": 1,
            "SCALING_FACTOR": scaling_factor,
            "KERNEL_NORMALIZER": kernel_normalizer
        } for scaling_factor, kernel_normalizer in generate_run_parameters(0.75, 1, 1, 1)
    ]

    main(config, diffs)


if __name__ == "__main__":
    run_no_normalization_tests()
