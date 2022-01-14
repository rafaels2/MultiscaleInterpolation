from collections import namedtuple
import Experiment
from Config.Config import config
from Config.Options import options
from ParamFit import fit_moving_and_quasi
from Tools.Utils import set_output_directory

Combination = namedtuple("Combination", ["method", "secondary"])
NUMBER_OF_SCALES = 5


def main():
    base_config = {
        "MANIFOLD": options.get_option("manifold", "euclidean")(),
        "SECONDARY_MANIFOLD": options.get_option("manifold", "rotations")(),
        "ORIGINAL_FUNCTION": options.get_option("original_function", "euclidean"),
        "NUMBER_OF_SCALES": 1,
        "SCALING_FACTOR": 0.75,
        "EXECUTION_NAME": "comparison_between_quasi_and_manifold_projection",
        "TEST_FILL_DISTANCE": 0.08
    }

    config.set_base_config(base_config)
    config.renew()

    methods = {
        # "Quasi-interpolation with quadratic reproduction": Combination("moving", None),
        # "Quasi-interpolation with constant reproduction": Combination("quasi", None),
        "Manifold projection with quadratic reproduction": Combination("projection", "quasi")
    }

    diffs = list()

    for method_name, method in methods.items():
        current_diff = dict()
        current_diff["MSE_LABEL"] = f"{method_name}"
        current_diff["SCALED_INTERPOLATION_METHOD"] = method.method
        current_diff["SECONDARY_SCALED_INTERPOLATION_METHOD"] = method.secondary
        for iteration in range(1, NUMBER_OF_SCALES + 1):
            current_diff["NAME"] = f"{method_name}_{iteration}"
            current_diff["SCALING_FACTOR_POWER"] = iteration
            current_diff["SCALING_FACTOR"] = config.SCALING_FACTOR ** iteration
            diffs.append(current_diff.copy())

    with set_output_directory("results"):
        results = Experiment.run_all_experiments(diffs)
        fit_moving_and_quasi(results)


if __name__ == "__main__":
    main()
