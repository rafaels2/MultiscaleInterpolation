from cachetools import cached, LFUCache
from contextlib import contextmanager
from matplotlib import pyplot as plt
from numpy import linalg as la
from matplotlib import cm
import numpy as np
import os

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


def handle_boundaries(func, index, boundaries, x, y):
    if boundaries:
        index = list(index)
        # Mirror conditions
        old = index[:]
        for i in range(0, 2):
            if boundaries_type == "periodic":
                if index[i] < boundaries:
                    index[i] = x.shape[i] - (2 * boundaries) - index[i]
                if x.shape[i] - index[i] < boundaries:
                    index[i] = boundaries + x.shape[i] - index[i]
            elif boundaries_type == "mirror":
                if index[i] < boundaries:
                    index[i] = boundaries + (boundaries - index[i])
                elif x.shape[i] - index[i] < boundaries:
                    index[i] = 2 * (x.shape[i] - boundaries) - index[i]
            elif boundaries_type == "constant":
                if index[i] < boundaries:
                    index[i] = boundaries
                elif x.shape[i] - index[i] < boundaries:
                    index[i] = x.shape[i] - boundaries
        if old != index:
            print("changed: ", old, index)
        index = tuple(index)
    else:
        pass
        # print("no boundaries")
    return func(x[index], y[index])


def evaluate_on_grid(func, *args, points=None, boundaries=0, should_log=False):
    if points is not None:
        x, y = points
    else:
        x, y = generate_grid(*args, should_ravel=False)
    
    z = np.zeros(x.shape, dtype=object)
    print("Z shape", z.shape)

    for index in np.ndindex(x.shape):
        if index[1] == 0 and should_log:
            print("current percentage: ", index[0] / x.shape[0])
        z[index] = handle_boundaries(func, index, boundaries, x, y)

    return z


def plot_and_save(data, filename, title):
    plt.figure()
    plt.title(title)
    plt.imshow(data)
    cb = plt.colorbar()
    plt.savefig(filename)
    cb.remove()


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


@contextmanager
def set_output_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    last_cwd = os.getcwd()
    os.chdir(path)
    yield

    os.chdir(last_cwd)
    return
