import numpy as np
from retraction_pairs import PositiveNumbers, Circle

_SCALING_FACTOR = 0.8

def _original_function(x, y):
    phi = 5 * y + x
    return np.array([np.cos(phi), np.sin(phi)])


CONFIG = {
    "GRID_SIZE": 1.5,
    "BASE_RESOLUTION": 3,
    "PLOT_RESOLUTION_FACTOR": 2,
    "SCALE": 1,
    "NUMBER_OF_SCALES": 4,
    "TEST_SCALE": _SCALING_FACTOR / 2,
    "SCALING_FACTOR": _SCALING_FACTOR,
    "NAME": "temp",
    "OUTPUT_DIR": "results",
    "EXECUTION_NAME": "1-4 scales and single scales",
    "ORIGINAL_FUNCTION": _original_function,
    "MANIFOLD": Circle()
}


DIFFS = [
        {"NAME": "{}_scale".format(x), "NUMBER_OF_SCALES": x} for x in range(3, 4)
    ]

# DIFFS = [
#         {"NAME": "{}_scale".format(x), "NUMBER_OF_SCALES": x} for x in range(3, 4   )
#     ] + [
#         {"NAME": "single_scale_{}".format(i), 
#          "NUMBER_OF_SCALES": 1,
#          "SCALING_FACTOR": _SCALING_FACTOR ** i} for i in range(3, 4)
#     ]
