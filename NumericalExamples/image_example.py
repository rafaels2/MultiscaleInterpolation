import matplotlib.pyplot as plt

import Experiment
from Config.Config import config
from Config.Options import options
from OriginalFunction import generate_image_function
from Tools.Utils import set_output_directory
from os.path import join
import os


def main():
    print(os.getcwd())
    generate_image_function("iguana", join("..", "images", "iguana.jpg"))
    plt.set_cmap("gray")
    base_config = {
        "NUMBER_OF_SCALES": 4,
        "ORIGINAL_FUNCTION": options.get_option("original_function", "iguana"),
        "EXECUTION_NAME": f"Image",
        "DATA_SITES_GENERATION": "grid",
        "MANIFOLD": options.get_option("manifold", "numbers")(),
        "DATA_SITES_STORAGE": "grid",
        "GRID_BORDER": 0.05,
        "GRID_SIZE": 0.95,
        "BASE_SCALE": 0.2,
        "BASE_RESOLUTION": 2,
        "TEST_FILL_DISTANCE": 0.008,
        "SCALING_FACTOR": 0.6,
        "CMAP": "gray",
        "CB": False,
    }

    config.set_base_config(base_config)
    config.renew()

    Experiment.run_all_experiments([{"MSE_LABEL": "AnomalyDetection"}])


if __name__ == "__main__":
    with set_output_directory("results"):
        main()
