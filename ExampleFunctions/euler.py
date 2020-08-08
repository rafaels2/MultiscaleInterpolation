""" Rotations """
from Manifolds.RigidRotations import Rotation


def original_function(x, y):
    return Rotation.from_euler('xyz', [x/2, y/2, x*y/4]).as_matrix()

