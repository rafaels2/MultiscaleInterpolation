import numpy as np
from pykdtree.kdtree import KDTree
from copy import copy

from . import register_generation

from Config.Config import config
from .Grid import get_grid


@register_generation("thinning")
def thin(x_min, x_max, y_min, y_max, fill_distance):
    sequence = copy(config.SEQUENCE)
    tree = KDTree(sequence)
    grid = get_grid(x_min, x_max, y_min, y_max, fill_distance, should_ravel=True)
    indices = set(
        tree.query(np.array([[x, y]]), k=1)[1][0] for x, y in zip(*grid)
    )

    result = [sequence[index] for index in indices if index < sequence.shape[0]]

    return np.array(result)
