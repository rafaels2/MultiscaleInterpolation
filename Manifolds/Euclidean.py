import numpy as np
from numpy import linalg as la

from Config.Config import config
from Manifolds import register_manifold
from Manifolds.RealNumbers import RealNumbers
from Tools.Visualization import RotationVisualizer


@register_manifold("euclidean")
class Euclidean(RealNumbers):
    def __init__(self):
        super().__init__()
        self.dim = config.EUCLIDEAN_DIMENSION

    def _to_numbers(self, x):
        return la.norm(x)

    def zero_func(self, x_0, x_1):
        return np.ones(self.dim)

    def plot(self, data, title, filename, norm_visualization=False):
        centers = np.zeros_like(data, dtype=object)
        for index in np.ndindex(data.shape):
            centers[index] = np.array([index[0], index[1], 0])
        RotationVisualizer(data, centers).save(filename, title)
