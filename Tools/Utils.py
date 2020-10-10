import os
from contextlib import contextmanager

import numpy as np
from cachetools import cached, LFUCache
from matplotlib import pyplot as plt
from numpy import linalg as la

from InputInterface import ConfidenceError
from Tools.SamplingPoints import generate_grid, SubDomain

num_of_caches_g = 0


def generate_cache(maxsize=32):
    global num_of_caches_g
    num_of_caches_g += 1
    # print("New Cache! Now having {}".format(num_of_caches_g))
    return LFUCache(maxsize=maxsize)


def act_on_functions(action, a, b):
    @cached(cache=generate_cache(maxsize=100))
    def new_func(*args):
        return action(a(*args), b(*args))

    return new_func


def evaluate_on_grid(func, *args, points=None, should_log=False):
    # Only used in interpolation
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


def paint(data, centers):
    color_map = np.zeros((centers[:, 0].max()+1, centers[:, 1].max()+1))
    for index in np.ndindex(data.shape):
        color_map[tuple(centers[index])] = data[index]
    return color_map


def plot_and_save(data, title, filename, centers=None):
    if centers is not None:
        data = paint(data, centers)
    plt.figure()
    plt.title(title)
    plt.imshow(data)
    cb = plt.colorbar()
    plt.savefig(filename)
    cb.remove()


def plot_lines(lines, filename, title, xlabel, ylabel):
    fig = plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for line_name, line_dots in lines.items():
        plt.plot(line_dots, label=line_name)
    plt.legend(lines.keys())
    plt.savefig(filename)
    plt.close(fig)


def generate_kernel(rbf, scale=1):
    def kernel(x, y):
        # TODO: change to kernel(p). make sure to replace 2 to d
        ans = (1 / scale ** 2) * rbf(la.norm(x - y) / scale)
        return ans

    return kernel


def wendland(x):
    if x < 0:
        raise ValueError("x should be > 0, not {}".format(x))
    if x > 1:
        return 0
    else:
        return (1 + (4 * x)) * ((1 - x) ** 4)


def calculate_max_derivative(original_function, confidence, grid_params, manifold):
    def derivative(x, y):
        # TODO: change (x, y) to p and write the derivatives in a more generic way.
        # delta = grid_params.mesh_norm / 2
        center = np.array([1, 1])
        evaluations_around = list()
        for _index in np.ndindex((3, 3)):
            current_index = 3 * (_index - center)
            if current_index[0] == 0 and current_index[1] == 0:
                continue
            try:
                evaluations_around.append(
                    manifold.distance(original_function(x + current_index[0], y + current_index[1]),
                                      original_function(x, y)) / la.norm(current_index, 2))
            except ConfidenceError:
                print("Skipped confidence")
                continue

        return max(evaluations_around)

    evaluation, centers = SubDomain(confidence, 1, derivative, grid_params).evaluation

    result = np.zeros_like(evaluation, dtype=np.float32)
    for index in np.ndindex(result.shape):
        result[index] = evaluation[index]

    return result


@contextmanager
def set_output_directory(path):
    """ 
    This is the easy implementation. 
    The correct one should not change the working directory,
    but keep it as a variable.
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    last_cwd = os.getcwd()
    os.chdir(path)
    yield

    os.chdir(last_cwd)
    return
