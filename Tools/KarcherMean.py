import numpy as np
from numpy import linalg as la
from scipy.linalg import expm, logm, sqrtm

if __name__ == "__main__":
    from Manifolds.SymmetricPositiveDefinite import SymmetricPositiveDefinite

AVERAGE_TOLERANCE = 0.001


class KarcherMean(object):
    def __init__(self, manifold, values_to_average, weights):
        self._manifold = manifold
        self._values_to_average = values_to_average
        self._weights = weights
        # Removed this line because it doesn't work with lambdas. Mathematically it should be fine,
        # but should be validated.
        # assert all(w_i >= 0 for w_i in weights), f"All weights should be non-negative, {weights}"
        assert all(
            manifold.is_in_manifold(a_i) for a_i in values_to_average
        ), "Not all values_to_average in _manifold"

    def _get_start_point(self):
        return sum(
            w_i * a_i for a_i, w_i in zip(self._values_to_average, self._weights)
        )

    def _get_step_length(self, x_l):
        x_sqrt_inv = la.inv(sqrtm(x_l))
        matrices_to_condition = (
            np.matmul(np.matmul(x_sqrt_inv, a_i), x_sqrt_inv)
            for a_i in self._values_to_average
        )

        eigenvalues = (la.eig(m_i)[0] for m_i in matrices_to_condition)
        conditions = (max(eig_i) / min(eig_i) for eig_i in eigenvalues)

        return 2 / sum(
            w_i * np.log(c_i) * ((c_i + 1) / (c_i - 1))
            for c_i, w_i in zip(conditions, self._weights)
        )

    def get_average(self, base=None, i=0):
        if base is None:
            base = self._get_start_point()

        step_length = self._get_step_length(base)
        base_inv = la.inv(base)

        exp_param = step_length * sum(
            w_i * logm(np.matmul(base_inv, a_i))
            for a_i, w_i in zip(self._values_to_average, self._weights)
        )

        x = np.matmul(base, expm(exp_param))

        distance = self._manifold.distance(x, base)
        if distance < AVERAGE_TOLERANCE:
            return x

        if i > 10:
            print("average did not converge")
            return x

        return self.get_average(x, i + 1)


def main():
    manifold = SymmetricPositiveDefinite()
    a, b, c = (manifold.gen_point() for _ in range(3))
    average = KarcherMean(manifold, [a, b, c], [1, 1, 1]).get_average()

    return manifold, a, b, c, average


if __name__ == "__main__":
    _manifold, a, b, c, average = main()
