import numpy as np


def original_function(x, y):
    # TODO: add check if function returns a valid manifold point.
    z = (np.abs(np.cos(7*y)) + 0.1) * np.exp(-x**2 -y**2) * (5 * np.eye(3) + np.array([[np.sin(5 * y), y, x*y], [0, 0, y ** 2],[0,0,0]]))
    return (z + np.transpose(z))
