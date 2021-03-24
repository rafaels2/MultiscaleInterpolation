import numpy as np

from Tools.SamplingPoints import Grid, generate_grid


def calculate_max_derivative(original_function, grid_params, manifold):
    def derivative(x, y):
        delta = grid_params.mesh_norm / 2
        evaluations_around = [
            original_function(x + (delta / np.sqrt(2)), y + (delta / np.sqrt(2))),
            original_function(x, y + delta),
            original_function(x, y - delta),
            original_function(x + (delta / np.sqrt(2)), y - (delta / np.sqrt(2))),
            original_function(x + delta, y),
            original_function(x - (delta / np.sqrt(2)), y + (delta / np.sqrt(2))),
            original_function(x - delta, y),
            original_function(x - (delta / np.sqrt(2)), y - (delta / np.sqrt(2))),
        ]
        f_0 = original_function(x, y)

        return max(
            [
                manifold.distance(direction, f_0) / delta
                for direction in evaluations_around
            ]
        )

    evaluation = Grid(1, derivative, grid_params).evaluation

    result = np.zeros_like(evaluation, dtype=np.float32)
    for index in np.ndindex(result.shape):
        result[index] = evaluation[index]

    return result


def evaluate_on_grid(func, *args, points=None, should_log=False):
    if points is not None:
        x, y = points
    else:
        x, y = generate_grid(*args, should_ravel=False)

    z = np.zeros(x.shape, dtype=object)
    print("Z shape", z.shape)

    for index in np.ndindex(x.shape):
        if index[1] == 0 and should_log:
            print("current percentage: ", index[0] / x.shape[0])
        z[index] = func(x[index], y[index])

    return z
