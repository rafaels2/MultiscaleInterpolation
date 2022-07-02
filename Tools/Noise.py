import numpy as np

from Config.Options import options

register_noise = options.get_type_register("noise")


@register_noise("none")
def none(func):
    return func


@register_noise("rotation_gaussian_noise")
def rotation_gaussian_noiser(func):
    def noised_func(*args):
        result = func(*args)
        return rotational_noise(result)
    return noised_func


def rotational_noise(result):
    manifold = options.get_option("manifold", "rotations")()
    noise = np.random.normal(0, 0.1, 3)
    tangent_noise = np.array([
        [0, noise[0], noise[1]],
        [0, 0, noise[2]],
        [0, 0, 0]
    ])
    tangent_noise = tangent_noise - np.transpose(tangent_noise)
    return manifold.exp(result, tangent_noise)
