from Manifolds.RigidRotations import RigidRotations, Rotation
from ApproximationMethods.Quasi import Quasi, QuasiNoNormalization

_SCALING_FACTOR = 0.75


def _original_function(x, y):
    return Rotation.from_euler('xyz', [x/2, y/2, x*y/4]).as_matrix()


CONFIG = {
    "GRID_SIZE": 1.5,
    "BASE_RESOLUTION": 2,
    "PLOT_RESOLUTION_FACTOR": 2,
    "SCALE": 1,
    "NUMBER_OF_SCALES": 4,
    "TEST_MESH_NORM": 0.1,
    "SCALING_FACTOR": _SCALING_FACTOR,
    "NAME": "temp",
    "OUTPUT_DIR": "results",
    "EXECUTION_NAME": "RotationsKarcher",
    "ORIGINAL_FUNCTION": _original_function,
    "MANIFOLD": RigidRotations(),
    "SCALED_INTERPOLATION_METHOD": QuasiNoNormalization,
    "NORM_VISUALIZATION": False,
    "IS_APPROXIMATING_ON_TANGENT": False,
    "MSE_LABEL":"Default Run",
    "IS_ADAPTIVE": False,
}


"""
DIFFS = [
        {"NAME": "{}_scale".format(x), "NUMBER_OF_SCALES": x} for x in range(3, 4)
    ]

"""
DIFFS = [
        {"MSE_LABEL": "Multiscale", "NAME": "{}_scale".format(x), "NUMBER_OF_SCALES": x} for x in range(1, 5)
    ] + [
        {"NAME": "single_scale_{}".format(i), 
         "MSE_LABEL": "Single scale",
         "NUMBER_OF_SCALES": 1,
         "SCALING_FACTOR": _SCALING_FACTOR ** i} for i in range(1, 5)
    ]

