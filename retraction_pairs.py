import numpy as np
from abc import abstractmethod



class AbstractManifold(object):
    @abstractmethod
    def exp(x, y):
        pass

    @abstractmethod
    def log(x, y):
        pass

    @abstractmethod
    def calculate_error(x, y):
        pass

    @abstractmethod
    def plot(x):
        pass

"""
S2 retraction pairs
"""
def exp(x, y):
    z = x + y
    return z / np.norm(z, ord=2)


def log(x, y):
    return ((y / np.dot(x, y)) - x)