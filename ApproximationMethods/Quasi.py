import numpy as np
from cachetools import cached
from numpy import linalg as la
from Tools.Utils import generate_grid, generate_kernel, evaluate_on_grid, generate_cache


def _calculate_phi(kernel, point):
    @cached(cache=generate_cache(maxsize=10))
    def phi(x, y):
        vector = np.array([x, y])
        return kernel(vector, point)
    return phi


def _interpolate(manifold, original_function, points, phis, hx, radius_in_index, min_value,
    is_approximating_on_tangent):
    values_at_points = evaluate_on_grid(original_function, points=points)

    @cached(cache=generate_cache(maxsize=10))
    def interpolant(x, y):
        x_0 = (x - min_value) / hx
        y_0 = (y - min_value) / hx
        values_to_average = list()
        weights = list()
        normalizer = 0

        for indx in np.ndindex((2 * radius_in_index + 2, 2 * radius_in_index + 2)):
            x_i = int(x_0 - radius_in_index - 1+ indx[0])
            y_i = int(y_0 - radius_in_index - 1 + indx[1])

            if any([x_i < 0, y_i <0, x_i >= phis.shape[0], y_i >= phis.shape[1]]):
                continue 

            current_phi_value = phis[y_i, x_i](x, y)
            values_to_average.append(values_at_points[y_i, x_i])
            weights.append(current_phi_value)
            normalizer += current_phi_value

        if normalizer == 0:
            normalizer = 0.00001

        weights = [w_i / normalizer for w_i in weights]
        if is_approximating_on_tangent:
            return sum(w_i * x_i for w_i, x_i in zip(weights, values_to_average))

        return manifold.average(values_to_average, weights)
    return interpolant


def quasi_scaled_interpolation(manifold, scale, original_function, \
                               grid_resolution, grid_size, rbf, is_approximating_on_tangent):
    x, y = generate_grid(grid_size, grid_resolution, scale, should_ravel=False)
    kernel = generate_kernel(rbf, scale)

    hx = (2 * grid_size / x.shape[0])
    radius_in_index = int(np.ceil(scale / hx))

    phis = list()
    for i in range(x.shape[0]):
        current_phis = list()
        for j in range(x.shape[1]):
            current_phis.append(_calculate_phi(kernel, np.array([x[i, j], y[i, j]])))
        phis.append(current_phis)
    phis = np.array(phis)

    interpolant = _interpolate(
        manifold,
        original_function,
        (x, y),
        phis,
        hx,
        radius_in_index,
        -grid_size,
        is_approximating_on_tangent
    )

    return interpolant


def main():
    pass


if __name__ == "__main__":
    main()
