import numpy as np
import os
import pickle as pkl
from matplotlib import pyplot as plt
import scipy.optimize
from collections import namedtuple
import argparse

FunctionData = namedtuple('FunctionData', ['mesh_norm', 'error'])
DIR = 'fit_results'


def plot_comparison(func, x_orig, y_orig, params, title):
    y_new = [func(x, *params) for x in x_orig]
    plt.figure()
    plt.plot(x_orig, y_orig, label='original')
    plt.plot(x_orig, y_new, label='fit')
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
    with open(filename, 'rb') as f:
        return pkl.load(f)


def fit_multi_scale(results):
    for i, name in enumerate(results['mses'].keys()):
        mses = results['mses'][name]
        x_orig = list(range(1, len(mses) + 1))
        (a, b), _ = scipy.optimize.curve_fit(
            _multi_linear, x_orig, mses, p0=[1, 1]
        )
        mu = results['mus'][i]
        plot_comparison(_multi_linear, x_orig, mses, (a, b),
                        f'Multi Scale Fit mu {mu}')
        yield a, b


def fit_mus(mus, param_b):
    # log_b = [np.log(b) for b in param_b]
    (const, curve), _ = scipy.optimize.curve_fit(
        _multi_linear, mus, param_b, p0=[1, 1]
    )
    plot_comparison(_multi_linear, mus, param_b, (const, curve), 'Mu Fit')
    return const, curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    args = parser.parse_args()

    experiment_results = pkl_load(args.filename)
    experiment_results['mus'] = experiment_results['mus'][::5]

    param_a = list()
    param_b = list()

    for a, b in fit_multi_scale(experiment_results):
        param_a.append(a)
        param_b.append(b)

    print(f'Average of a: {np.average(param_a)}, '
          f'stderr a: {np.std(param_a)}')

    const, curve = fit_mus(experiment_results['mus'], param_b)

    print(f'Const: {const}'
          f'Curve: {curve}')


if __name__ == '__main__':
    # single, multi = main()
    main()
