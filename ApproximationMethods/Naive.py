""" Currently deprecated """
import numpy as np
from numpy import linalg as la

from ApproximationMethods.ApproximationMethod import ApproximationMethod
from DataSites.Storage.Grid import Grid
from Tools.Utils import generate_kernel
from . import register_approximation_method


@register_approximation_method("naive")
class Naive(ApproximationMethod):
    def __init__(
        self,
        manifold,
        original_function,
        grid_parameters,
        rbf,
        scale,
        is_approximating_on_tangent,
    ):
        super().__init__(manifold, original_function, grid_parameters, rbf)
        self.is_approximating_on_tangent = is_approximating_on_tangent
        rbf_radius = scale

        # TODO: this is not OK with the new design
        self._grid = Grid(
            rbf_radius,
            original_function,
            grid_parameters[0][1],
            phi_generator=self._calculate_phi,
        )
        self._kernel = generate_kernel(self._rbf, rbf_radius)
        self.values_at_points = self._grid.evaluation.ravel()
        points_as_vectors = [
            np.array([x, y]) for x, y in zip(self._grid.x.ravel(), self._grid.y.ravel())
        ]
        kernel = np.array(
            [
                [self._kernel(x_i, x_j) for x_j in points_as_vectors]
                for x_i in points_as_vectors
            ]
        )
        self.coefficients = np.matmul(la.inv(kernel), self.values_at_points)

    def approximation(self, x, y):
        return sum(
            b_j * self._kernel(np.array([x, y]), np.array([x_j, y_j]))
            for b_j, x_j, y_j in zip(
                self.coefficients, self._grid.x.ravel(), self._grid.y.ravel()
            )
        )
