import numpy as np


def original_function(x, y):
    # TODO: add check if function returns a valid manifold point.
    z = (0.3 * np.abs(np.cos(2*y)) + 0.6) * np.exp(-x**2 -y**2) * (5 * np.eye(3) + np.array([[np.sin(5 * y), y, x*y], [0, 0, y ** 2],[0,0,0]])) + 0.3 * np.eye(3)
    return (z + np.transpose(z))


def ____original_function(x, y):
    # TODO: add check if function returns a valid manifold point.
    z = (np.abs(np.cos(7*y)) + 0.1) * np.exp(-x**2 -y**2) * (5 * np.eye(3) + np.array([[np.sin(5 * y), y, x*y], [0, 0, y ** 2],[0,0,0]])) 
    return (z + np.transpose(z))

