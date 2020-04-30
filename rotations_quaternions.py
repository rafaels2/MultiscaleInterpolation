""" Rotations """
import numpy as np
from Manifolds.RigidRotations import RigidRotations, Quaternion, Rotation


r = RigidRotations()
I = np.eye(3)


def _original_function(x, y):
    q = 0.5 * Quaternion(x, y, x * y, x**2 + y ** 2)
    return r.exp(I, q)
