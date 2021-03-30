import numpy as np
import os
import pickle as pkl
from matplotlib import pyplot as plt
import scipy.optimize
from collections import namedtuple
import argparse

FunctionData = namedtuple("FunctionData", ["mesh_norm", "error"])
DIR = "fit_results"


def sort_points(x_orig, y_orig):
    zipped = [(x, y) for x, y in zip(x_orig, y_orig)]
    zipped.sort(key=lambda x: x[0])
    x_list = [p[0] for p in zipped]
    y_list = [p[1] for p in zipped]
    return x_list, y_list


def plot_comparison(func, x_orig, y_orig, params, title, xlabel="log(h_X)", ylabel="log(Error)"):
    y_new = [func(x, *params) for x in x_orig]
    plt.figure()
    plt.plot(x_orig, y_orig, label="original")
    if xlabel == "iteration":
        plt.xticks(x_orig)
    plt.plot(x_orig, y_new, label="fit")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(DIR, f"{title.replace(' ', '_')}.png"))
    plt.show(block=False)


def _multi_linear(x, *params):
    (a, b) = params
    return b * x + a


def _quasi_error(x, *params):
    c_big, nu = params
    return np.log(c_big * (np.exp(x) ** nu))


def pkl_load(filename):
    with open(filename, "rb") as f:
        return pkl.load(f)


def fit_multi_scale(results, keyword="multiscale"):
    for i, name in enumerate(results["mses"].keys()):
        mses = results["mses"][name]
        # if "multiscale" in name:
        x_orig = list(range(1, len(mses) + 1))
        # else:
        #     x_orig = results["mesh_norms"][name]
        (a, b), _ = scipy.optimize.curve_fit(_multi_linear, x_orig, mses, p0=[1, 1])
        # mu = results['mus'][i]
        plot_comparison(_multi_linear, x_orig, mses, (a, b), f"Multi Scale Quasi Interpolation Fit {name}", "iteration")
        yield a, b, (keyword in name)


def fit_mus(mus, param_b, debug=False):
    # log_b = [np.log(b) for b in param_b]
    (const, curve), _ = scipy.optimize.curve_fit(_multi_linear, mus, param_b, p0=[1, 1])
    if not debug:
        plot_comparison(_multi_linear, mus, param_b, (const, curve), "mu param fit", "log(mu)", "multiscale fit slope")
    return const, curve


def fit_multi_and_single(experiment_results, keyword="multiscale"):
    param_a = list()
    param_b = list()
    multiscale_param_a = list()
    multiscale_param_b = list()

    for a, b, is_multiscale in fit_multi_scale(experiment_results):
        if is_multiscale:
            multiscale_param_a.append(a)
            multiscale_param_b.append(b)
        else:
            param_a.append(a)
            param_b.append(b)

    print(f"Average of a: {np.average(param_a)}, " f"stderr a: {np.std(param_a)}")

    const, curve = fit_mus([np.log(m) for m in experiment_results['mus']], param_b)
    # x_orig, y_orig = sort_points(param_a, multiscale_param_a)
    # const, curve = fit_mus(x_orig, y_orig)

    print(f"Const: {const}" f"Curve: {curve}")

    return param_a, multiscale_param_a, const, curve


def fit_moving_and_quasi(experiment_results):
    for a, b, is_moving in fit_multi_scale(experiment_results, keyword="moving"):
        if is_moving:
            moving_a = a
            moving_b = b
        else:
            quasi_a = a
            quasi_b = b

    print(f'moving slope is {moving_b}\n'
          f'quasi slope is {quasi_b}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    args = parser.parse_args()

    experiment_results = pkl_load(args.filename)
    experiment_results["mus"] = experiment_results["mus"][::5]

    return fit_multi_and_single(experiment_results)


if __name__ == "__main__":
    # single, multi = main()
    main()
