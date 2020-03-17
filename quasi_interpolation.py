import numpy as np
from numpy import linalg as la
from utils import generate_grid, generate_kernel, evaluate_original_on_points, \
    wendland, mse, sum_functions_list, div_functions, generate_cache
from cachetools import cached, LFUCache


def const(x, y):
    return 1


def _calculate_phi(kernel, point):
    @cached(cache=generate_cache(maxsize=10))
    def phi(x, y):
        vector = np.array([x, y])
        return kernel(vector, point)
    return phi


def _interpolate(original_function, points, phis, hx, radius_in_index, min_value):
    values_at_points = evaluate_original_on_points(original_function, points)

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
        
        return float(averages / normalizer)
    return interpolant


def quasi_scaled_interpolation(scale, original_function, grid_resolution, grid_size, rbf):
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

    interpolant = _interpolate(original_function, (x, y), phis, hx, radius_in_index, -grid_size)

    return interpolant


def main():
    """
    This main tests the assumptions on the quasi interpolation theory
    """
    rbf = wendland
    x, y = generate_grid(5, 3, 1)
    phis = [_calculate_phi(rbf, 1, np.array([x_i, y_i]), polynom) for x_i, y_i in zip(x, y)]

    q = polynom
    
    def test_function(x_test, y_test):
        return sum(_calculate_phi(rbf, 1, np.array([x_i, y_i]), polynom)(x_test, y_test) for x_i, y_i in zip(x, y))

    x_test, y_test = generate_grid(5, 3, 2)

    print("mse was: {}".format(mse(test_function, q, x_test, y_test)))


if __name__ == "__main__":
    main()
