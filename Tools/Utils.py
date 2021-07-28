import os
from collections import namedtuple
from contextlib import contextmanager

from cachetools import cached, LFUCache

from matplotlib import cm
from matplotlib import pyplot as plt

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


def plot_and_save(data, title, filename):
    plt.figure()
    # plt.title(title)
    fig = plt.imshow(data)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    cb = plt.colorbar()
    plt.savefig(filename)
    cb.remove()


def plot_lines(x_values, y_values, filename, title, x_label, y_label):
    fig = plt.figure()
    # plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    for line_name in y_values.keys():
        if x_values is not None:
            plt.plot(x_values[line_name], y_values[line_name], label=line_name)
        else:
            _x_values = [x + 1 for x in range(len(y_values[line_name]))]
            plt.plot(_x_values, y_values[line_name], label=line_name)
            plt.xticks(_x_values)
    plt.legend(y_values.keys())
    plt.savefig(filename, bbox_inches='tight')
    if ".svg" in filename:
        plt.savefig(f"{filename[:-4]}.png", bbox_inches='tight')
    plt.close(fig)


def generate_kernel(rbf, scale=1):
    def kernel(x, y):
        # ans = (1 / scale ** 2) * rbf(la.norm(x - y) / scale)
        # ans = (1/scale) * rbf(la.norm(x-y) * 1.25 / scale)
        ans = rbf(1.25 * la.norm(x - y) / scale)
        return ans

    return kernel


def wendland_3_1(x):
    if x < 0:
        raise ValueError("x should be > 0, not {}".format(x))
    if x > 1:
        return 0
    else:
        return (1 + (4 * x)) * ((1 - x) ** 4)


def wendland_3_0(x):
    if x < 0:
        raise ValueError("x should be > 0, not {}".format(x))
    if x > 1:
        return 0
    else:
        return (1 - x) ** 2


def wendland_3_2(x):
    if x < 0:
        raise ValueError("x should be > 0, not {}".format(x))
    if x > 1:
        return 0
    else:
        return (35 * (x ** 2) + 18 * x + 3) * (1 - x) ** 6


def wendland_1_0(x):
    if x < 0:
        raise ValueError("x should be > 0, not {}".format(x))
    if x > 1:
        return 0
    else:
        return 1 - x


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
