import argparse
import importlib

import Interpolation
from ApproximationMethods.AdaptiveQuasi import AdaptiveQuasi
from ApproximationMethods.MovingLeastSquares import MovingLeastSquares
from ApproximationMethods.Naive import Naive
from ApproximationMethods.NoNormalization import NoNormalization
from ApproximationMethods.Quasi import Quasi
from Config import CONFIG, _SCALING_FACTOR
from Tools.Utils import set_output_directory, wendland_3_0, wendland_3_1, wendland_3_2, wendland_1_0
from Manifolds import MANIFOLDS

METHODS = {'naive': Naive, 'quasi': Quasi, 'adaptive': AdaptiveQuasi, 'moving': MovingLeastSquares}


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
    parser.add_argument('-e', '--execution-name', type=str, default='NoName')
    parser.add_argument('-a', '--adaptive', action='store_true', help='is adaptive m0')
    parser.add_argument('-mt', '--method', choices=METHODS.keys(), default='quasi', help='approximation method')
    parser.add_argument('-ci', '--compare-to-interpolation', action='store_true')
    parser.add_argument('-dm', '--dont-multi', action='store_true')
    args = parser.parse_args()

    config = CONFIG.copy()
    config['ORIGINAL_FUNCTION'] = importlib.import_module(args.function).original_function
    config['MANIFOLD'] = MANIFOLDS[args.manifold]()
    config['IS_APPROXIMATING_ON_TANGENT'] = args.tangent_approximation
    config['NORM_VISUALIZATION'] = args.norm_visualization
    config['SCALING_FACTOR'] = args.scaling_factor
    is_tangent = "Tangent" if args.tangent_approximation else "Intrinsic"
    config['EXECUTION_NAME'] = "{}_{}".format(args.manifold, is_tangent)
    execution_name = args.execution_name if (args.execution_name != 'NoName') else "{}_{}".format(args.manifold, is_tangent)
    config['EXECUTION_NAME'] = execution_name
    config['IS_ADAPTIVE'] = args.adaptive
    config['SCALED_INTERPOLATION_METHOD'] = METHODS[args.method]

    if args.dont_multi:
        diffs = []
    else:
        diffs = [
            {
                "NAME":"multiscale",
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

    if args.compare_to_interpolation:
        print('Comparing')
        diffs_copy = diffs.copy()
        for item in diffs_copy:
            new_item = item.copy()
            new_item['NAME'] = '_'.join([item['NAME'], 'interpolation'])
            new_item['MSE_LABEL'] = ' '.join(['Interpolated', item['MSE_LABEL']])
            new_item['SCALED_INTERPOLATION_METHOD'] = Naive
            diffs.append(new_item)

    return config, diffs


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
        # "moving": MovingLeastSquares,
        # "quasi": Quasi,
        "no_normalization": NoNormalization
    }

    for name, wendland in wendlands.items():
        for method_name, method in methods.items():
            for diff in old_diffs:
                current_diff = diff.copy()
                current_diff["NAME"] = f"{method_name}_{name}_{current_diff['NAME']}"
                current_diff["MSE_LABEL"] = f"{method_name}_{name}_{current_diff['MSE_LABEL']}"
                current_diff["RBF"] = wendland
                current_diff["SCALED_INTERPOLATION_METHOD"] = method
                diffs.append(current_diff)

    return config, diffs


def main():
    config, diffs = parse_arguments()
    original_function = config['ORIGINAL_FUNCTION']
    output_dir = CONFIG["OUTPUT_DIR"]

    config, diffs = run_different_rbfs(config, diffs)

    with set_output_directory(output_dir):
        results = Interpolation.run_all_experiments(config, diffs, original_function)


if __name__ == "__main__":
    main()
