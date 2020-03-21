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


def generate_original_function():
    @cached(cache={})
    def original_function(x, y):
        return np.sin(5*x)*np.cos(4*y)

    return original_function


def evaluate_on_grid(func, *args):
    x, y = generate_grid(*args, should_ravel=False)
    z = np.zeros(x.shape, dtype=object)
    print("Z shape", z.shape)

    for index in np.ndindex(x.shape):
        if index[1] == 0:
            print(index)
        z[index] = func(x[index], y[index])

    return x, y, z


def plot_contour(ax, func, *args):
    x, y, z = evalueat_on_grid(func, *args)
    ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    return z


def plot_and_save(data, filename, title):
    plt.figure()
    plt.title(title)
    plt.imshow(data)
    cb = plt.colorbar()
    plt.savefig(filename)
    cb.remove()


def generate_grid(grid_size, resolution, scale=1, should_ravel=True):
    print("creating a grid", 2 * resolution / scale)
    y = np.linspace(-grid_size, grid_size, int(2 * resolution / scale))
    x = np.linspace(-grid_size, grid_size, int(2 * resolution / scale))
    x_matrix, y_matrix = np.meshgrid(x, y)
    if should_ravel:
        return x_matrix.ravel(), y_matrix.ravel()
    else:
         return x_matrix, y_matrix


def generate_kernel(rbf, scale=1):
    def kernel(x, y):
        ans = (1 / scale ** 2) * rbf(la.norm(x-y) / scale)
        return ans

    return kernel


def mse(func_a, func_b, x, y):
    errors = np.zeros(x.shape)
    for index in np.ndindex(x.shape):
        errors[index] = np.square(func_a(x[index], y[index]) - func_b(x[index], y[index]))
    return errors.mean()


def run_on_array(function, x, y):
    return np.array([function(x[index], y[index]) for index in np.ndindex(x.shape)])


def act_on_functions(action, a, b):
    @cached(cache=generate_cache(maxsize=100))
    def new_func(*args):
        return action(a(*args), b(*args))
    return new_func


def sum_functions(a, b):
    def new_func(*args):
        # TODO: Use the P transform
        return a(*args) + b(*args)

    return new_func


def sum_functions_list(funcs):

    def new_func(*args):
        return sum(func(*args) for func in funcs)

    return new_func


def div_functions(a, b):
    @cached(cache=generate_cache(maxsize=100))
    def new_func(*args):
        return a(*args) / b(*args)

    return new_func


def sub_functions(a, b):
    def new_func(*args):
        # TODO: Use the Q transform
        return a(*args) - b(*args)

    return new_func


def zero_func(*args):
    return 0


def wendland(x):
    if x < 0:
        raise ValueError("x should be > 0, not {}".format(x))
    if x > 1:
        return 0
    else:
        return (1 + (4 * x)) * ((1 - x) ** 4)


def evaluate_original_on_points(original_function, points):
    print("started interpolation")
    x, y = points
    print("x shape {}".format(x.shape))
    values_at_points = np.zeros_like(x, dtype=object)
    for index in np.ndindex(x.shape):
        x_i = x[index]
        y_i = y[index]
        values_at_points[index] = original_function(x_i, y_i) 
    return values_at_points


@contextmanager
def set_output_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    last_cwd = os.getcwd()
    os.chdir(path)
    yield

    os.chdir(last_cwd)
    return
