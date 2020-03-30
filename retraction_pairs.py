import numpy as np
from numpy import linalg as la
from abc import abstractmethod

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

ALLOWED_AVERAGING_ERROR = 0.05


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

    @abstractmethod
    def _get_geodetic_line(self, x, y):
        pass

    def _geodetic_average_two_points(self, x, y, w_x, w_y):
        if not w_y:
            return x
        if not w_x:
            return y
        ratio = w_x / (w_x + w_y)
        line = self._get_geodetic_line(x, y)
        return line(ratio)

    def _geodetic_average(self, values_to_average, weights):
        """
        This calculation is recursive because we always know to 
        calculate the geodetic average only for a couple of points.
        """
        average = self._geodetic_average_two_points(
            values_to_average[0],
            values_to_average[-1],
            weights[0],
            weights[-1]
        )
        
        values_to_average = values_to_average[:-1]
        new_weight = weights[0] + weights[-1]
        values_to_average[0] = average
        weights = weights[:-1]
        weights[0] = new_weight

        if len(values_to_average) == 1:
            return values_to_average[0]
        else:
            return self._geodetic_average(values_to_average, weights)

    def _karcher_mean(self, values_to_average, weights, base=None, iterations=0):
        if base is None:
            base = values_to_average[0]

        total_weight = 0
        average = 0

        for value, weight in zip(values_to_average, weights):
            average += weight * self.log(values_to_average[0], value)
            total_weight += weight

        new_base = self.exp(base, average / total_weight)
        print("new_base: ", new_base)
        if la.norm(self.log(base, new_base)) < ALLOWED_AVERAGING_ERROR:
            print("Calculated iterations: ", iterations)
            return new_base

        return self._karcher_mean(values_to_average, weights, base=new_base, iterations=(iterations+1))

    def average(self, values_to_average, weights):
        """
        This function is the last resort...
        We can make it iterative to get to better results
        """
        return self._karcher_mean(values_to_average, weights)


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
            return self.gen_point(theta)

        return line

    def average(self, values_to_average, weights):
        return self._geodetic_average(values_to_average, weights)


def main():
    m = Circle()
    a = m.gen_point(0)
    b = m.gen_point(2)
    e = m.gen_point(1)
    f = m.gen_point (0.5)
    s = m.average([a, b], [1, 1])
    print("s: ", np.arctan2(s[1], s[0]))
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
