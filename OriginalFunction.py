"""
List of functions to examine
"""
# TODO: add option to get an image.
# TODO: add an option to get any data.
import numpy as np

from Config.Options import options

register_function = options.get_type_register("original_function")


@register_function("numbers")
def numbers(x, y):
    return np.sin(4 * x) * np.cos(5 * y)


@register_function("numbers_gauss")
def numbers_gauss(x, y):
    return 5 * (np.exp(-(x ** 2) - y ** 2))


@register_function("one")
def one(*_):
    return 1


@register_function("numbers_sin")
def numbers_sin(x, y):
    return np.sin(2 * (x + 0.5)) * np.cos((3 * (y + 0.5)))
