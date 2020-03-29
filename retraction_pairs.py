import numpy as np
from numpy import linalg as la
from abc import abstractmethod

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


class AbstractManifold(object):
    @abstractmethod
    def exp(self, x, y):
        pass

    @abstractmethod
    def log(self, x, y):
        pass

    def calculate_error(self, x, y):
        error = np.zeros_like(x, dtype=np.float32)
        for index in np.ndindex(x.shape):
            error[index] = la.norm(self.log(x[index], y[index]))
        return error

    @abstractmethod
    def _visualize(self, plt, data):
        pass    

    def plot(self, data, title, filename):
        plt.figure()
        plt.title(title)
        cb = self._visualize(plt, data)
        plt.savefig(filename)
        cb.remove()

    def zero_func(self, x_0, x_1):
        return 0

    def average(self, values_to_average, weights):
        total_weight = 0
        average = 0

        for value, weight in zip(values_to_average, weights):
            average += weight * self.log(values_to_average[0], value)
            total_weight += weight

        return self.exp(values_to_average[0], average / total_weight)


class PositiveNumbers(AbstractManifold):
    def exp(self, x, y):
        return x ** y

    def log(self, x, y):
        # TODO: Think if we want to change to log(1+x)
        if x == 1:
            epsilon = 0.00001
        else:
            epsilon = 0
        return np.log(y) / (np.log(x) + epsilon)

    def _visualize(self, plt, data):
        plt.imshow(data)
        cb = plt.colorbar()
        return cb

    def zero_func(self, x_0, x_1):
        return 2


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
        return ((y / np.abs(inner_product)) - x)

    def _visualize(self, plt, data):
        """
        WARNING! this usage of arctan can be missleading - it can choose the 
        incorrect brach.
        I guess that plotting the log can be better.
        """
        visualization = np.zeros_like(data, dtype=np.float32)
        for index in np.ndindex(visualization.shape):
            x = data[index]
            visualization[index] = np.arctan2(x[1], x[0])
        plt.imshow(visualization)
        cb = plt.colorbar()
        return cb

    def gen_point(self, phi):
        return np.array([np.cos(phi), np.sin(phi)])

    def zero_func(self, x_0, x_1):
        return np.array([0, 1])


def main():
    m = Circle()
    a = m.gen_point(0)
    b = m.gen_point(2)
    e = m.gen_point(1)
    f = m.gen_point (0.5)
    c = m.log(a, b)
    g = m.log(a, e)
    h = m.log(a, f)

    d = m.exp(a, c)
    i = m.exp(a, c + g + h)
    j = m.exp(d, g + h)

    print("d-b: ", d-b)
    print("a^(c+g+h) - (a^c)^(g+h)", i-j)


if __name__ == "__main__":
    main()
