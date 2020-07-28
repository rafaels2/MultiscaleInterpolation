import numpy as np

from Manifolds.SymmetricPositiveDefinite import SymmetricPositiveDefinite
from Manifolds.RealNumbers import RealNumbers
from Manifolds.Circle import Circle


r = RigidRotations()
I = np.eye(3)


def _original_function(x, y):
    q = 0.5 * Quaternion(x, y, x * y, x**2 + y ** 2)
    return r.exp(I, q)


"""
def _original_function(x, y):
    q = Quaternion(np.sin(5 * x - 4 *y), np.exp(-x**2-y**2), x*y, x**2 + y**2)
    return r.exp(I, q)
"""


"""
# SPD
def _original_function(x, y):
    # TODO: add check if function returns a valid manifold point.
    z = (np.abs(np.cos(7*y)) + 0.1) * np.exp(-x**2 -y**2) * (5 * np.eye(3) + np.array([[np.sin(5 * y), y, x*y], [0, 0, y ** 2],[0,0,0]]))
    return (z + np.transpose(z))
"""

"""
# Real Numbers
def _original_function(x, y):
    return np.sin(5*x) * np.cos(4*y) + np.sin(7*x*y) - x ** 2 - y ** 2
"""

"""
Circle
def _original_function(x, y):
    phi = (np.pi / 2) * np.exp(-x**2 - y**2)
    return np.array([np.cos(phi), np.sin(phi)])
"""
"""
def _original_function(x, y):
    q = Quaternion(np.sin(5 * x - 4 *y), np.exp(-x**2-y**2), x*y, x**2 + y**2)
    return r.exp(I, q)
"""


"""
# SPD
def _original_function(x, y):
    # TODO: add check if function returns a valid manifold point.
    z = (np.abs(np.cos(7*y)) + 0.1) * np.exp(-x**2 -y**2) * (5 * np.eye(3) + np.array([[np.sin(5 * y), y, x*y], [0, 0, y ** 2],[0,0,0]]))
    return (z + np.transpose(z))
"""

"""
# Real Numbers
def _original_function(x, y):
    return np.sin(5*x) * np.cos(4*y) + np.sin(7*x*y) - x ** 2 - y ** 2
"""

"""
Circle
def _original_function(x, y):
    phi = (np.pi / 2) * np.exp(-x**2 - y**2)
    return np.array([np.cos(phi), np.sin(phi)])
"""
