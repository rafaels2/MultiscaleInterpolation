import numpy as np

from . import register_generation


@register_generation("grid")
def get_grid(x_min, x_max, y_min, y_max, fill_distance, should_ravel=False):
    # TODO: generalize the approximation Domain. from 2D to any nD.
    y = np.linspace(y_min, y_max, int((y_max - y_min) / fill_distance))
    x = np.linspace(x_min, x_max, int((x_max - x_min) / fill_distance))
    x_matrix, y_matrix = np.meshgrid(x, y)
    if should_ravel:
        return x_matrix.ravel(), y_matrix.ravel()
    else:
        return x_matrix, y_matrix


def generate_grid(grid_size, resolution, scale=1, should_ravel=True):
    print("creating a grid", 2 * resolution / scale)
    y = np.linspace(-grid_size, grid_size, int(2 * resolution / scale))
    x = np.linspace(-grid_size, grid_size, int(2 * resolution / scale))
    x_matrix, y_matrix = np.meshgrid(x, y)
    if should_ravel:
        return x_matrix.ravel(), y_matrix.ravel()
    else:
        return x_matrix, y_matrix
