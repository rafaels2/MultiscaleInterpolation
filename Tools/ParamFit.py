import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize
from collections import namedtuple
from Config import _SCALING_FACTOR

FunctionData = namedtuple('FunctionData', ['mesh_norm', 'error'])

RESULTS = {
    'numbers_gauss':
        {
            'quasi':
                {
                    'mses': [2.8570535, 2.3486903, 1.8128927, 1.2604101, 0.698276, 0.13016611, -0.4409449],
                    'mesh_norms': [
                        -0.2586945355785286, -0.5463766080303095, -0.8340586804820905, -1.1217407529338714,
                        -1.4094228253856522, -1.6971048978374332, -1.984786970289214]
                },
            'multi':
                {
                    'mses': [2.8570535, 1.1803906, -0.5231396, -1.5175103, -2.16627, -3.117656, -4.3528185],
                    'mesh_norms': [
                        -0.2586945355785286, -0.5463766080303095, -0.8340586804820905, -1.1217407529338714,
                        -1.4094228253856522, -1.6971048978374332, -1.984786970289214
                    ]
                }
        },
    # 'numbers_one':
    #     {
    #         'quasi':
    #             {
    #                 'mesh_norms'
    #             },
    #         'multi':
    #             {
    #
    #             }
    #     }
}


def plot_comparison(func, x_orig, y_orig, params, title):
    y_new = [func(x, *params) for x in x_orig]
    plt.figure()
    plt.plot(x_orig, y_orig, label='original')
    plt.plot(x_orig, y_new, label='fit')
    plt.title(title)
    plt.show(block=False)


def _multi_linear(x, *params):
    (a, b) = params
    return a * x + b


def run_results():
    for func, result in RESULTS.copy().items():
        _result = result.copy()
        y_orig = [np.log(x) for x in _result['multi']['mses']]
        x_orig = list(range(1, len(y_orig) + 1))
        (A, B), _ = scipy.optimize.curve_fit(
            _multi_linear, x_orig, y_orig, p0=[1, 1]
        )
        plot_comparison(_multi_linear, x_orig, y_orig, (A, B), 'Single Scale Fit')

    print(A, B)
# noinspection PyTypeChecker
def _run_results():
    for func, result in RESULTS.copy().items():
        _result = result.copy()
        x_orig = _result['quasi']['mesh_norms']
        y_orig = _result['quasi']['mses']
        (quasi_const, convergence), _ = scipy.optimize.curve_fit(
            _quasi_error, x_orig, y_orig, p0=[1, 1]
        )
        # noinspection PyTypeChecker
        RESULTS[func]['quasi']['params'] = {
            "convergence": convergence,
            "quasi_const": quasi_const,
        }
        plot_comparison(_quasi_error, x_orig, y_orig, (quasi_const, convergence), 'Single Scale Fit')

        multi_mses = _result['multi']['mses']
        multi_differences = []
        for i in range(len(multi_mses) - 1):
            multi_differences.append(2 * multi_mses[i + 1] - 2 * multi_mses[i])

        RESULTS[func]['multi']['differences'] = multi_differences

    nu = np.average([
        result['quasi']['params']['convergence'] for result in RESULTS.values()
    ])

    for func, result in RESULTS.copy().items():
        x_orig = _result['multi']['mesh_norms'][1:]
        y_orig = _result['multi']['differences']
        _result = result.copy()
        # import ipdb
        # ipdb.set_trace()

        func = get_multi_error(nu)
        (c_big, power_function), _ = scipy.optimize.curve_fit(
            func, x_orig, y_orig, p0=(1, 1),
        bounds=([0 , 0], [1000,1000]))

        plot_comparison(func, x_orig, y_orig, (c_big, power_function), 'Multi Scale Fit')
        # noinspection PyTypeChecker
        RESULTS[func]['multi']['params'] = {
            'power_function': power_function,
            'c_big': c_big,
            'norm_from_quasi': _result['quasi']['params']['quasi_const'] / c_big
        }


def _quasi_error(x, *params):
    c_big, nu = params
    return np.log(c_big * (np.exp(x) ** nu))


def get_multi_error(nu):
    def _multi_error(x, *params):
        c_big, c = params
        return np.log((2 ** 3) * (c_big * (np.exp(x) ** nu) + (_SCALING_FACTOR ** (2 * 3)) * c))
    return _multi_error


def main():
    error_multi = [x for x in [
        3.9069011, 3.751316, 3.4572103, 3.0197046, 2.3680975, 1.4181411, 0.122505374, -1.4758203, -2.851662]]
    error_single = [x for x in [
        3.9069011, 3.7754338, 3.5624442, 3.2720752, 2.8900316, 2.4361544, 1.9334888, 1.400371, 0.848413]]
    # error_single = [0.8626434, -0.17491123, -1.2536587, -2.3585966, -3.4808002]
    mesh_norms = [x for x in [
        -0.2586945355785286, -0.5463766080303095, -0.8340586804820905,
        -1.1217407529338714, -1.4094228253856522, -1.6971048978374332,
        -1.984786970289214, -2.2724690427409953, -2.560151115192776
    ]]
    # mesh_norms = [-0.2586945355785286, -0.5463766080303095, -0.8340586804820905, -1.1217407529338714, -1.4094228253856522]
    result_single = scipy.optimize.curve_fit(_quasi_error, mesh_norms, error_single, p0=np.array([1, 1]))
    # result_multi = scipy.optimize.curve_fit(_multi_error, mesh_norms, error_multi, p0=np.array([70, 1, -1]),bounds=(
    #     [65, 0.5, -100], [100, 1.5, 0]
    # ))

    ydata = [_quasi_error(x, *tuple(result_single[0])) for x in mesh_norms]
    plt.plot(mesh_norms, ydata, label='fit')
    plt.plot(mesh_norms, error_single, label='approx')
    plt.legend()
    plt.show()
    print(result_single)
    return result_single


if __name__ == '__main__':
    # single, multi = main()
    run_results()
