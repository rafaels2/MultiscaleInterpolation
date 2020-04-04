import numpy as np
from Manifolds.SymmetricPositiveDefinite import SymmetricPositiveDefinite
from ApproximationMethods.Quasi import quasi_scaled_interpolation

_SCALING_FACTOR = 0.8

def _original_function(x, y):
    z = (5 + x) * np.eye(3) + np.array([[np.sin(5 * y), y, x*y], [0, 0, y ** 2],[0,0,0]])
    return (z + np.transpose(z))


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
    "MANIFOLD": SymmetricPositiveDefinite(),
    "SCALED_INTERPOLATION_METHOD": quasi_scaled_interpolation,
    "NORM_VISUALIZATION": True,
    "IS_APPROXIMATING_ON_TANGENT": False
}


DIFFS = [
        {"NAME": "{}_scale".format(x), "NUMBER_OF_SCALES": x} for x in range(1, 2)
    ]

# DIFFS = [
#         {"NAME": "{}_scale".format(x), "NUMBER_OF_SCALES": x} for x in range(1, 5)
#     ] + [
#         {"NAME": "single_scale_{}".format(i), 
#          "NUMBER_OF_SCALES": 1,
#          "SCALING_FACTOR": _SCALING_FACTOR ** i} for i in range(1, 5)
#     ]
