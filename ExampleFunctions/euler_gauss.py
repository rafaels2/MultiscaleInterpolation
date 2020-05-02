""" Rotations """
import numpy as np
from Manifolds.RigidRotations import RigidRotations, Quaternion, Rotation


def original_function(x, y):
    return Rotation.from_euler('xyz', [0.5 * (1-np.exp(-x**2)), 0.5 * (1 - np.exp(-y**2)), 0.2 * np.cos(2*x*y)]).as_matrix()

