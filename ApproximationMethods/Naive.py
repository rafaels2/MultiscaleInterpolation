""" Currently deprecated """
import numpy as np
from numpy import linalg as la

from ApproximationMethods.ApproximationMethod import ApproximationMethod
from Tools.GridUtils import evaluate_on_grid
from Tools.SamplingPoints import Grid, generate_grid
from Tools.Utils import generate_kernel


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


def _interpolate(phi, original_function, points):
    """
    Generating I_Xf(x) for the given kernel and points
    :param phi: Kernel
    :param original_function:
    :param points:
    :return:
    """
    points_as_vectors, values_at_points = evaluate_on_grid(
        original_function, points=points
    )
    kernel = np.array(
        [[phi(x_i, x_j) for x_j in points_as_vectors] for x_i in points_as_vectors]
    )
    coefficients = np.matmul(la.inv(kernel), values_at_points)
    print(kernel)

    def interpolant(x, y):
        return sum(
            b_j * phi(np.array([x, y]), x_j)
            for b_j, x_j in zip(coefficients, points_as_vectors)
        )

    return interpolant


def naive_scaled_interpolation(
    scale, original_function, grid_resolution, grid_size, rbf
):
    x, y = generate_grid(grid_size, grid_resolution, scale)
    phi = generate_kernel(rbf, scale)
    return _interpolate(phi, original_function, (x, y))
