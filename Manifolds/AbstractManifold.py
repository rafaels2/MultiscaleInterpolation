import numpy as np
from numpy import linalg as la
from abc import abstractmethod

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

ALLOWED_AVERAGING_ERROR = 10 ** -3
SYMMETRIC_ERROR = 10 ** -5


class AbstractManifold(object):
    @abstractmethod
    def exp(self, x, y):
        pass

    @abstractmethod
    def log(self, x, y):
        pass

    def distance(self, x, y):
        return la.norm(self.log(x, y))

    def calculate_error(self, x, y):
        error = np.zeros_like(x, dtype=np.float32)
        for index in np.ndindex(x.shape):
            # Relative Error
            error[index] = self.distance(x[index], y[index])
        return error

    @abstractmethod
    def _to_numbers(self, x):
        pass

    def _visualize(self, plt, data):
        visualization = np.zeros_like(data, dtype=np.float32)
        for index in np.ndindex(visualization.shape):
            x = data[index]
            visualization[index] = self._to_numbers(x)
        fig = plt.imshow(visualization)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        cb = plt.colorbar()
        return cb

    def plot(self, data, title, filename, **kwargs):
        self.fig = plt.figure()
        plt.title(title)
        cb = self._visualize(plt, data)
        plt.savefig(filename)
        cb.remove()
        plt.close(self.fig)

    def zero_func(self, x_0, x_1):
        return 0

    @abstractmethod
    def _get_geodetic_line(self, x, y):
        pass

    def _geodesic_average_two_points(self, x, y, w_x, w_y):
        if not w_y:
            return x
        if not w_x:
            return y
        ratio = w_x / (w_x + w_y)
        line = self._get_geodetic_line(x, y)
        return line(ratio)

    def _geodesic_average(self, values_to_average, weights):
        """
        This calculation is recursive because we always know to 
        calculate the geodetic average only for a couple of points.
        """
        average = self._geodesic_average_two_points(
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
            return self._geodesic_average(values_to_average, weights)

    @abstractmethod
    def _karcher_mean(self, values_to_average, weights, base=None, iterations=0):
        pass

    def average(self, values_to_average, weights):
        """
        This function is the last resort...
        We can make it iterative to get to better results
        """
        return self._geodesic_average(values_to_average, weights)


if __name__ == "__main__":
    # main()
    pass
