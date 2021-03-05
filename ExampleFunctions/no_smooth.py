import numpy as np

CONST = 5


def original_function(x, y):
    if np.floor(CONST * y / np.pi) % 2 == 0:
        if np.floor(CONST * x / np.pi) % 2 == 0:
            return 0.5 * np.sin(CONST * y) * np.cos(x * CONST)
        else:
            return -0.5 * np.sin(CONST * y) * np.cos(x * CONST)
    else:
        if np.floor(CONST * x / np.pi + np.pi / 2) % 2 == 0:
            return 0.5 * np.sin(-CONST * y) * np.cos(x * CONST)
        else:
            return -0.5 * np.sin(-CONST * y) * np.cos(x * CONST)
