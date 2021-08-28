import Experiment
from Config.Config import config
from Config.Options import options
from Tools.Utils import set_output_directory


def main():
    base_config = {
        "NUMBER_OF_SCALES": 4,
        "ORIGINAL_FUNCTION": options.get_option("original_function", "image"),
        "EXECUTION_NAME": f"anomaly_detection",
        "DATA_SITES_GENERATION": "halton",
        "MANIFOLD": options.get_option("manifold", "numbers")(),
        "DATA_SITES_STORAGE": "kd-tree",
        "GRID_BORDER": 0.1,
        "GRID_SIZE": 0.9,
        # "BASE_SCALE": 0.1,
        # "TEST_FILL_DISTANCE": 0.002
    }

    config.set_base_config(base_config)
    config.renew()

    Experiment.run_all_experiments([{"MSE_LABEL": "AnomalyDetection"}])


if __name__ == "__main__":
    with set_output_directory("results"):
        main()
