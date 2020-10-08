import argparse
import importlib

import numpy as np
import Interpolation
from Config import CONFIG, _SCALING_FACTOR
from DataSetParser import NRRDParser
from InputInterface import DTMRIDataSet
from Tools.Utils import set_output_directory, wendland
from Manifolds import MANIFOLDS


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


def main():
    config, diffs = parse_arguments()
    rbf = wendland
    original_function = config['ORIGINAL_FUNCTION']
    output_dir = CONFIG["OUTPUT_DIR"]

    with set_output_directory(output_dir):
        _ = Interpolation.run_all_experiments(config, diffs, rbf, original_function)


def get_nrrd_as_function(path):
    parser = NRRDParser(path)
    data, resolution = parser.parse()
    print(f"Resolution is: {resolution}")
    original_function = DTMRIDataSet(data, np.array([1, 1, 1]), resolution)
    return original_function


if __name__ == "__main__":
    main()
    # orig = get_nrrd_as_function('input_data\\dt-helix.nhdr')
    # print(orig(np.array([0.03, 0.03, 0.03])))
