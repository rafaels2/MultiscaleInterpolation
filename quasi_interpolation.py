import numpy as np
from cachetools import cached
from numpy import linalg as la
from utils import generate_grid, generate_kernel, evaluate_on_grid, generate_cache


def _calculate_phi(kernel, point):
    @cached(cache=generate_cache(maxsize=10))
    def phi(x, y):
        vector = np.array([x, y])
        return kernel(vector, point)
    return phi


def _interpolate(manifold, original_function, points, phis, hx, radius_in_index, min_value, boundaries):
    print('boundaries: ', boundaries)
    values_at_points = evaluate_on_grid(
        original_function,
        points=points,
        boundaries=boundaries
    )

    @cached(cache=generate_cache(maxsize=10))
    def interpolant(x, y):
        x_0 = (x - min_value) / hx
        y_0 = (y - min_value) / hx
        averages = 0
        normalizer = 0
        
        """
        TODO:
            1. averages should be calculated with:
                1.1 The Q, P formula; or
                1.2 Least squares (I  guess it's more accurate)
            2. phis are the lambdas
        """
        for indx in np.ndindex((2 * radius_in_index + 2, 2 * radius_in_index + 2)):
            x_i = int(x_0 - radius_in_index - 1+ indx[0])
            y_i = int(y_0 - radius_in_index - 1 + indx[1])
            
            if any([x_i < 0, y_i <0, x_i >= phis.shape[0], y_i >= phis.shape[1]]):
                continue 
            
            current_phi_value = phis[y_i, x_i](x, y)
            averages += current_phi_value * values_at_points[y_i, x_i] 
            normalizer += current_phi_value
        
        
        if normalizer == 0:
            normalizer = 0.00001
        
        return (averages / normalizer)
    return interpolant


def quasi_scaled_interpolation(manifold, scale, original_function, grid_resolution, grid_size, rbf):
    # boundaries = scale
    boundaries = scale
    x, y = generate_grid(grid_size + boundaries, grid_resolution, scale, should_ravel=False)
    kernel = generate_kernel(rbf, scale)

    hx = (2 * grid_size / x.shape[0])
    radius_in_index = int(np.ceil(scale / hx))
    boundaries_in_index = int(np.ceil(boundaries / hx))

    phis = list()
    for i in range(x.shape[0]):
        current_phis = list()
        for j in range(x.shape[1]):
            current_phis.append(_calculate_phi(kernel, np.array([x[i, j], y[i, j]])))
        phis.append(current_phis)
    phis = np.array(phis)

    interpolant = _interpolate(manifold, original_function, (x, y), phis, hx, radius_in_index, -grid_size, boundaries=boundaries_in_index)

    return interpolant


def main():
    pass


if __name__ == "__main__":
    main()
