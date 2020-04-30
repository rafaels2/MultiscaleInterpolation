from Manifolds.RigidRotations import RigidRotations, Quaternion, Rotation


def original_function(x, y):
    return Rotation.from_euler('xyz', [x/2, y/2, x*y/4]).as_matrix()

