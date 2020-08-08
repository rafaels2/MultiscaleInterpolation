""" Rotations """
import numpy as np
from Manifolds.RigidRotations import RigidRotations, Quaternion, Rotation


R = RigidRotations()
ID_MATRIX = np.eye(3)


def _original_function(x, y):
    q = 0.5 * Quaternion(x, y, x * y, x**2 + y ** 2)
    return R.exp(ID_MATRIX, q)
