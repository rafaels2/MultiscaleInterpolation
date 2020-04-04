import numpy as np
from numpy import linalg as la

from scipy.linalg import expm, logm, sqrtm
from sklearn.datasets import make_spd_matrix

from Tools.KarcherMean import KarcherMean
from Tools.Visualization import ElipsoidVisualizer

from .AbstractManifold import AbstractManifold

SYMMETRIC_ERROR = 10 ** -5


class SymmetricPositiveDefinite(AbstractManifold):
    def __init__(self, dim=3):
        super().__init__()
        self.dim = dim

    def is_in_manifold(self, x):
        return all([
            (la.norm(x - np.transpose(x)) < SYMMETRIC_ERROR),
            (all(x > 0 for x in la.eig(x)[0]))
        ])

    def exp(self, x, y):
        sqrt_x = sqrtm(x)
        exp_param = self._calculate_log_param(x, y)
        return np.matmul(np.matmul(sqrt_x, expm(exp_param)), sqrt_x)

    def old_log(self, x, y):
        sqrt_x = sqrtm(x)
        inv_sqrt_x = la.inv(sqrt_x) 
        return sqrt_x * logm(inv_sqrt_x * y * inv_sqrt_x) * sqrt_x

    def _calculate_log_param(self, x, y):
        sqrt_x = sqrtm(x)
        inv_sqrt_x = la.inv(sqrt_x) 
        return np.matmul(np.matmul(inv_sqrt_x, y), inv_sqrt_x)
        
    def log(self, x, y):
        sqrt_x = sqrtm(x)
        log_param = self._calculate_log_param(x, y)
        return np.matmul(np.matmul(sqrt_x, logm(log_param)), sqrt_x)

    def distance(self, x, y):
        log_param = self._calculate_log_param(x, y)
        return la.norm(logm(log_param))

    def _to_numbers(self, x):
        return la.norm(x, ord=2)

    def gen_point(self):
        return make_spd_matrix(self.dim)

    def zero_func(self, x_0, x_1):
        return np.eye(self.dim)

    def _get_geodetic_line(self, x, y):
        sqrt_x = sqrtm(x)
        log_param = self._calculate_log_param(x, y)

        def line(t):
            return sqrt_x * logm(t * log_param) * sqrt_x

        return line

    def _karcher_mean(self, values_to_average, weights, base=None):
        return KarcherMean(self, values_to_average, weights).get_average()

        return self._karcher_mean(values_to_average, weights, base=x)

    def average(self, values_to_average, weights):
        return self._karcher_mean(values_to_average, weights)

    def plot(self, data, title, filename, norm_visualization=False):
        if norm_visualization:
            return super().plot(data, title, filename)
        centers = np.zeros_like(data, dtype=object)
        for index in np.ndindex(data.shape):
            centers[index] = np.array([index[0], index[1], 0])
        print("start to visualize")
        ElipsoidVisualizer(data, centers).save(filename, title)


def main():
    m = SymmetricPositiveDefinite()
    a = m.gen_point()
    b = m.gen_point()
    e = m.gen_point()
    f = m.gen_point ()
    # s = m.average([a, b], [1, 1])
    # print("s: ", np.arctan2(s[1], s[0]))
    c = m.log(a, b)
    g = m.log(a, e)
    h = m.log(a, f)

    d = m.exp(a, c)
    i = m.exp(a, c + g + h)
    j = m.exp(d, g + h)

    for x in [d, i, j]:
        print("is in", m.is_in_manifold(x))

    print("d-b: ", m.distance(d, b))
    print("a^(c+g+h) - (a^c)^(g+h)", m.distance(i, j))


if __name__ == "__main__":
    main()
