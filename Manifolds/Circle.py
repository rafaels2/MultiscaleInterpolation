import numpy as np
from numpy import linalg as la

from .AbstractManifold import AbstractManifold
from . import register_manifold


def gen_point(phi):
    return np.array([np.cos(phi), np.sin(phi)])


@register_manifold("circle")
class Circle(AbstractManifold):
    """
    S2 retraction pairs
    """

    def exp(self, x, y):
        z = x + y
        return z / la.norm(z, ord=2)

    def log(self, x, y):
        inner_product = np.inner(x, y)
        if inner_product == 0:
            inner_product = 0.00001
        return (y / np.abs(inner_product)) - x

    def _to_numbers(self, x):
        """
        WARNING! this usage of arctan can be misleading - it can choose the
        incorrect branch.
        I guess that plotting the log can be better.
        """
        return np.arctan2(x[1], x[0])

    def zero_func(self, x_0, x_1):
        return np.array([0, 1])

    def _get_geodetic_line(self, x, y):
        theta_x = np.arctan2(x[1], x[0])
        theta_y = np.arctan2(y[1], y[0])
        if max(theta_x, theta_y) - min(theta_x, theta_y) >= np.pi:
            if theta_x > theta_y:
                theta_x -= 2 * np.pi
            else:
                theta_y -= 2 * np.pi

        def line(t):
            theta = theta_x + ((theta_y - theta_x) * (1 - t))
            return gen_point(theta)

        return line
