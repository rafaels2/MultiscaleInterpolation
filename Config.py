import numpy as np
from Manifolds.SymmetricPositiveDefinite import SymmetricPositiveDefinite
from Manifolds.RealNumbers import RealNumbers
from Manifolds.Circle import Circle
from Manifolds.RigidRotations import RigidRotations, Quaternion
from ApproximationMethods.Quasi import Quasi

_SCALING_FACTOR = 0.7

r = RigidRotations()
I = np.eye(3)

def _original_function(x, y):
    q = Quaternion(np.sin(5 * x - 4 *y), np.exp(-x**2-y**2), x*y, x**2 + y**2)
    return r.exp(I, q)


"""
SPD
def _original_function(x, y):
    # TODO: add check if function returns a valid manifold point.
    z = (np.abs(np.cos(7*y)) + 0.1) * np.exp(-x**2 -y**2) * (5 * np.eye(3) + np.array([[np.sin(5 * y), y, x*y], [0, 0, y ** 2],[0,0,0]]))
    return (z + np.transpose(z))
"""

# """
"""Real Numbers"""
def _original_function(x, y):
    return np.sin(5*x) * np.cos(4*y) + np.sin(7*x*y) - x ** 2 - y ** 2
# """

"""
Circle
def _original_function(x, y):
    phi = (np.pi / 2) * np.exp(-x**2 - y**2)
    return np.array([np.cos(phi), np.sin(phi)])
"""


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
    "EXECUTION_NAME": "RotationsTangent",
    "ORIGINAL_FUNCTION": _original_function,
    "MANIFOLD": RigidRotations(),
    "SCALED_INTERPOLATION_METHOD": Quasi,
    "NORM_VISUALIZATION": True,
    "IS_APPROXIMATING_ON_TANGENT": True,
    "MSE_LABEL":"Default Run",
}


"""
DIFFS = [
        {"NAME": "{}_scale".format(x), "NUMBER_OF_SCALES": x} for x in range(1, 5)
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
