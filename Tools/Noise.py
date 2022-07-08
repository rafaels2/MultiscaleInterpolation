import numpy as np
from cachetools import cached

from Config.Config import config
from Config.Options import options
from Tools.Utils import generate_cache

register_noise = options.get_type_register("noise")


@register_noise("none")
def none(func):
    return func


@register_noise("rotation_gaussian_noise")
def rotation_gaussian_noiser(func):
    @cached(cache=generate_cache(maxsize=10000))
    def noised_func(*args):
        result = func(*args)
        return rotational_noise(result)

    return noised_func


def rotational_noise(result):
    manifold = options.get_option("manifold", "rotations")()
    noise = np.random.normal(0, config.NOISE_SIGMA, 3)
    tangent_noise = np.array([[0, noise[0], noise[1]], [0, 0, noise[2]], [0, 0, 0]])
    tangent_noise = tangent_noise - np.transpose(tangent_noise)
    return manifold.exp(result, tangent_noise)
