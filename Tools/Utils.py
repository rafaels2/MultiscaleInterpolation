import os
from contextlib import contextmanager

from cachetools import cached, LFUCache

from matplotlib import pyplot as plt

from numpy import linalg as la

from Config.Config import config

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
    print(f"Saving {filename}")
    plt.figure()
    # plt.title(title)
    fig = plt.imshow(data)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if config.CB:
        cb = plt.colorbar()
    plt.savefig(filename, bbox_inches='tight')
    if config.CB:
        cb.remove()


def plot_lines(x_values, y_values, filename, title, x_label, y_label):
    fig = plt.figure()
    # plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    for line_name in y_values.keys():
        if x_values is not None:
            if isinstance(x_values, dict):
                plt.plot(x_values[line_name], y_values[line_name],'o--' , label=line_name)
            else:
                plt.plot(x_values, y_values[line_name], 'o--', label=line_name)
        else:
            _x_values = [x + 1 for x in range(len(y_values[line_name]))]
            plt.plot(_x_values, y_values[line_name], label=line_name)
            plt.xticks(_x_values)
    plt.legend(y_values.keys())
    plt.savefig(filename, bbox_inches="tight")
    if ".svg" in filename:
        plt.savefig(f"{filename[:-4]}.png", bbox_inches="tight")
    plt.close(fig)


def generate_kernel(rbf, scale=1):
    def kernel(x, y):
        ans = rbf(la.norm(x - y) / scale)
        return ans

    return kernel


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


def config_plt(_plt):
    _plt.rc("font", size=20)  # controls default text size
    _plt.rc("axes", titlesize=20)  # fontsize of the title
    _plt.rc("axes", labelsize=20)  # fontsize of the x and y labels
    _plt.rc("xtick", labelsize=20)  # fontsize of the x tick labels
    _plt.rc("ytick", labelsize=20)  # fontsize of the y tick labels
    _plt.rc("legend", fontsize=20)  # fontsize of the legend
    _plt.rc("axes", labelsize=20)
    _plt.tight_layout()
