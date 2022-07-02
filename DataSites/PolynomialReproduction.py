"""
Generate coefficients for the quasi-interpolation that promise polynomial reproduction.
Based on H. Wendland's paper on local polynomial reproduction.
"""
import os
import pickle as pkl

import numpy as np
from numpy import linalg as la

condition_g = list()


class PolynomialReproduction(object):
    def __init__(self, grid, filename="cache.pkl"):
        self.grid = grid

        # Coefficients for the linear problem. Currently for quadratic reproduction.
        self.polynomial_coefficients = [
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 0], [1, 0]]),
            np.array([[0, 1], [0, 0]]),
            # TODO: Make this configurable
            np.array([[0, 0], [0, 1]]),
            np.array([[0, 0, 1], [0, 0, 0]]),
            np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
        ]
        self._filename = filename

        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self._lambdas = pkl.load(f)
        else:
            self._lambdas = {}

    def _calculate(self, x, y):
        points_in_radius = [points for points in self.grid.points_in_radius(x, y)]
        if len(points_in_radius) <= 3:
            import ipdb

            ipdb.set_trace()
        polynomials_in_radius = np.array(
            [
                [
                    np.polynomial.polynomial.polyval2d(x_i.x, x_i.y, c_j)
                    for c_j in self.polynomial_coefficients
                ]
                for x_i in points_in_radius
            ]
        )

        kernel = np.diag([x_i.phi(x, y) for x_i in points_in_radius])

        polynomials_at_point = np.array(
            [
                [np.polynomial.polynomial.polyval2d(x, y, c_j)]
                for c_j in self.polynomial_coefficients
            ]
        )

        try:
            # Solve the linear problem of polynomial reproduction
            to_inv = np.matmul(
                np.matmul(np.transpose(polynomials_in_radius), kernel),
                polynomials_in_radius,
            )

            # This is the condition number of the problem
            cond = la.cond(to_inv)
            print(f"Condition {cond}")
            condition_g.append(cond)
            return 2 * np.matmul(la.inv(to_inv), polynomials_at_point)
        except la.LinAlgError as e:
            print(f"Singular")
            return np.matmul(to_inv, polynomials_at_point)

    def calculate(self, x, y):
        value = self._lambdas.get((x, y), None)
        if value is None:
            print(f"calculating {x}, {y}")
            value = self._calculate(x, y)
            self._lambdas[(x, y)] = value

        return value

    def update(self):
        with open(self._filename, "wb") as f:
            pkl.dump(self._lambdas, f, protocol=pkl.HIGHEST_PROTOCOL)

    def weight_for_grid(self, x_j, y_j):
        """Get a(x, y) coefficient for the quasi-interpolation {sum a(p)f(p_i)}"""

        def weight(x, y):
            # TODO: Debug! why did i need to transpose?
            return np.inner(
                np.transpose(self.calculate(x, y)),
                np.array(
                    [
                        np.polynomial.polynomial.polyval2d(x_j, y_j, c_j)
                        for c_j in self.polynomial_coefficients
                    ]
                ),
            )

        return weight
