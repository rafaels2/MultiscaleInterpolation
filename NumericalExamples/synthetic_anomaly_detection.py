import Experiment
from Config.Config import config
from Config.Options import options
from Tools.Utils import set_output_directory


def main():
    base_config = {
        "NUMBER_OF_SCALES": 5,
        "ORIGINAL_FUNCTION": options.get_option(
            "original_function", "anomaly_synthetic"
        ),
        "EXECUTION_NAME": f"anomaly_detection",
        "DATA_SITES_GENERATION": "halton",
        "MANIFOLD": options.get_option("manifold", "numbers")(),
        "DATA_SITES_STORAGE": "kd-tree",
    }

    config.set_base_config(base_config)
    config.renew()

    Experiment.run_all_experiments([{"MSE_LABEL": "AnomalyDetection"}])


if __name__ == "__main__":
    with set_output_directory("results"):
        main()
