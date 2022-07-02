"""
Fit utils to validate theory
"""
import time

import numpy as np
import os
import pickle as pkl
from matplotlib import pyplot as plt
import scipy.optimize
from collections import namedtuple
import argparse

import Tools.Utils
from Tools.Utils import set_output_directory

Tools.Utils.config_plt(plt)

RunData = namedtuple(
    "RunData",
    [
        "reproduction_degree",
        "wendland_index",
        "scaling_factor",
        "function_name",
        "is_multiscale",
        "error_values",
        "mesh_norms",
    ],
)
DIR = "fit_results"


def parse_results(results):
    """Parse results.pkl"""
    tags = results["mses"].keys()
    return [
        RunData(
            0 if "quasi" in approximation_type.lower() else 1,
            wendland_index,
            mu.split("_")[1],
            function_name,
            "multiscale" in scale_tag,
            results["mses"][tag],
            results["mesh_norms"][tag],
        )
        for tag, (
            approximation_type,
            wendland_index,
            mu,
            function_name,
            scale_tag,
        ) in ((tag, tag.split("__")) for tag in tags)
    ]


def sort_points(x_orig, y_orig):
    zipped = [(x, y) for x, y in zip(x_orig, y_orig)]
    zipped.sort(key=lambda x: x[0])
    x_list = [p[0] for p in zipped]
    y_list = [p[1] for p in zipped]
    return x_list, y_list


def plot_comparison(
    func,
    x_orig,
    y_orig,
    params,
    title,
    xlabel="log(h_X)",
    y_err=None,
    ylabel="log(Error)",
    is_fit=False,
):
    y_new = [func(x, *params) for x in x_orig]
    plt.figure()
    plt.plot(x_orig, y_new, "--", label="fit")
    plt.grid()
    if y_err is not None:
        plt.scatter(x_orig, y_orig)
        plt.errorbar(x_orig, y_orig, yerr=y_err, fmt="o")
    elif is_fit:
        plt.scatter(x_orig, y_orig, marker="*")
    else:
        plt.plot(x_orig, y_orig, "o", label="original")
        plt.legend()
    if xlabel == "iteration":
        plt.xticks(x_orig)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    path = "{}_{}".format(DIR, time.strftime("%Y%m%d__%H%M%S"))

    with set_output_directory(path):
        plt.savefig(f"{title.replace(' ', '_')}.png", bbox_inches="tight")


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
        name = name[len("multiscale ") :]
        x_orig = list(range(1, len(mses) + 1))
        title = f"Multiscale quasi interpolation fit {name.lower()}"
        xlabel = "iteration"
        is_multi = True
        # else:
        #     x_orig = results["mesh_norms"][name]
        #     title = f"Quasi-interpolation fit for {name.lower()}"
        #     xlabel = "log$(h_X)$"
        #     is_multi = False
        (a, b), pcov = scipy.optimize.curve_fit(_multi_linear, x_orig, mses, p0=[1, 1])
        stdev = np.sqrt(np.diag(pcov))
        # mu = results['mus'][i]
        plot_comparison(_multi_linear, x_orig, mses, (a, b), title, xlabel, is_fit=True)
        yield a, b, stdev[1], is_multi


def fit_mus(mus, param_b, b_err, debug=False):
    # log_b = [np.log(b) for b in param_b]
    (const, curve), _ = scipy.optimize.curve_fit(
        _multi_linear, mus, param_b, sigma=b_err, p0=[1, 1]
    )
    if not debug:
        plot_comparison(
            _multi_linear,
            mus,
            param_b,
            (const, curve),
            "mu param fit",
            "log($\mu$)",
            ylabel="multiscale fit slope",
            y_err=b_err,
            is_fit=True,
        )
    return const, curve


def fit_multi_and_single(experiment_results, keyword="multiscale"):
    param_a = list()
    param_b = list()
    multiscale_param_a = list()
    multiscale_param_b = list()
    multiscale_param_b_err = list()

    for a, b, b_err, is_multiscale in fit_multi_scale(experiment_results):
        if is_multiscale:
            multiscale_param_a.append(a)
            multiscale_param_b.append(b)
            multiscale_param_b_err.append(b_err)
        else:
            param_a.append(a)
            param_b.append(b)

    print(
        f"Average of a: {np.average(multiscale_param_a)}, "
        f"stderr a: {np.std(multiscale_param_a)}"
    )

    const, curve = fit_mus(
        [np.log(m) for m in experiment_results["mus"]],
        multiscale_param_b,
        multiscale_param_b_err,
    )
    # x_orig, y_orig = sort_points(param_a, multiscale_param_a)
    # const, curve = fit_mus(x_orig, y_orig)

    print(f"Const: {const}" f"Curve: {curve}")

    return param_a, multiscale_param_a, const, curve


def fit_single_scale(results, keyword):
    for i, name in enumerate(results["mses"].keys()):
        mses = results["mses"][name]
        x_orig = results["mesh_norms"][name]
        title = f"Quasi-interpolation fit for {name.lower()}"
        xlabel = "log$(h_X)$"
        is_keyword = keyword in name
        (a, b), _ = scipy.optimize.curve_fit(_multi_linear, x_orig, mses, p0=[1, 1])
        plot_comparison(_multi_linear, x_orig, mses, (a, b), title, xlabel)
        yield a, b, is_keyword


def fit_moving_and_quasi(experiment_results):
    for a, b, is_moving in fit_single_scale(experiment_results, keyword="Quadratic"):
        if is_moving:
            moving_b = b
        else:
            quasi_b = b

    print(f"moving slope is {moving_b}\n" f"quasi slope is {quasi_b}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    args = parser.parse_args()

    experiment_results = pkl_load(args.filename)
    experiment_results["mus"] = experiment_results["mus"][::4]

    # return fit_moving_and_quasi(experiment_results)
    return fit_multi_and_single(experiment_results)


if __name__ == "__main__":
    main()
    input()
