import os
from collections import namedtuple
from contextlib import contextmanager

from cachetools import cached, LFUCache

from matplotlib import cm
from matplotlib import pyplot as plt

import numpy as np
from numpy import linalg as la

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


def generate_grid(grid_size, resolution, scale=1, should_ravel=True):
    print("creating a grid", 2 * resolution / scale)
    y = np.linspace(-grid_size, grid_size, int(2 * resolution / scale))
    x = np.linspace(-grid_size, grid_size, int(2 * resolution / scale))
    x_matrix, y_matrix = np.meshgrid(x, y)
    if should_ravel:
        return x_matrix.ravel(), y_matrix.ravel()
    else:
         return x_matrix, y_matrix


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


def plot_and_save(data, title, filename):
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
        ans = (1 / scale ** 2) * rbf(la.norm(x-y) / scale)
        return ans

    return kernel


def wendland(x):
    if x < 0:
        raise ValueError("x should be > 0, not {}".format(x))
    if x > 1:
        return 0
    else:
        return (1 + (4 * x)) * ((1 - x) ** 4)


def calculate_max_derivative(original_function, grid_params, manifold):
    def derivative(x, y):
        delta = grid_params.mesh_norm / 2
        evaluations_around = [original_function(x + (delta / np.sqrt(2)), y + (delta / np.sqrt(2))),
                              original_function(x, y + delta),
                              original_function(x, y - delta),
                              original_function(x + (delta / np.sqrt(2)), y - (delta / np.sqrt(2))),
                              original_function(x + delta, y),
                              original_function(x - (delta / np.sqrt(2)), y + (delta / np.sqrt(2))),
                              original_function(x - delta, y),
                              original_function(x - (delta / np.sqrt(2)), y - (delta / np.sqrt(2)))]
        f_0 = original_function(x, y)

        return max([manifold.distance(direction, f_0)/delta for direction in evaluations_around])

    evaluation = Grid(1, derivative, grid_params).evaluation

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
