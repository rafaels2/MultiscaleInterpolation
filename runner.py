import argparse
import importlib

import Interpolation
from Config import CONFIG, DIFFS, SymmetricPositiveDefinite, RealNumbers, Circle, RigidRotations, _SCALING_FACTOR
from Tools.Utils import set_output_directory, wendland


MANIFOLDS = {
    'spd': SymmetricPositiveDefinite,
    'numbers': RealNumbers,
    'circle': Circle,
    'rotations': RigidRotations
}


def parse_arguments():
    parser = argparse.ArgumentParser("RBF Approximation Scrtipt")
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
    args = parser.parse_args()

    config = CONFIG.copy()
    config['ORIGINAL_FUNCTION'] = importlib.import_module(args.function).original_function
    config['MANIFOLD'] = MANIFOLDS[args.manifold]
    config['IS_APPROXIMATING_ON_TANGENT'] = args.tangent_approximation
    config['NORM_VISUALIZATION'] = args.norm_visualization
    config['SCALING_FACTOR'] = args.scaling_factor

    diffs = [
        {
            "NAME":"{}_scale".format(index),
            "NUMBER_OF_SCALES": index,
            "MSE_LABEL": "Multi Scale"
        } for index in range(args.base_index, args.base_index + args.number_of_scales)]

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
        results, interpolants = Interpolation.run_all_experiments(config, diffs, rbf, original_function)


if __name__ == "__main__":
    main()
