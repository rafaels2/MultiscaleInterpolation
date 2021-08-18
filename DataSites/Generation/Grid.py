import numpy as np

from . import register_generation


@register_generation("grid")
def get_grid(x_min, x_max, y_min, y_max, fill_distance, should_ravel=False):
    """
    Generate grid with the following properties.
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param fill_distance:
    :param should_ravel: Should return as a tuple of matrices x, y or as a single matrix [x, y] of two columns.
    :return: The grid according to should_ravel
    """
    # TODO: generalize the approximation Domain. from 2D to any nD.
    y = np.linspace(y_min, y_max, int(np.round((y_max - y_min) / fill_distance) + 1))
    x = np.linspace(x_min, x_max, int(np.round((x_max - x_min) / fill_distance) + 1))
    x_matrix, y_matrix = np.meshgrid(x, y)
    if should_ravel:
        return x_matrix.ravel(), y_matrix.ravel()
    else:
        return x_matrix, y_matrix
