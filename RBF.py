"""
Different phi_{d,k} wendland functions.
"""
from Config.Options import options

register_rbf = options.get_type_register("rbf")


def safe_rbf(func):
    def _rbf(x):
        if x < 0:
            raise ValueError("x should be > 0, not {}".format(x))
        if x > 1:
            return 0
        else:
            return func(x)

    return _rbf


@register_rbf("wendland_1_0")
@safe_rbf
def wendland_1_0(x):
    return 1 - x


@register_rbf("wendland_3_2")
@safe_rbf
def wendland_3_2(x):
    return (35 * (x ** 2) + 18 * x + 3) * (1 - x) ** 6


@register_rbf("wendland_3_0")
@safe_rbf
def wendland_3_0(x):
    return (1 - x) ** 2


@register_rbf("wendland_3_1")
@safe_rbf
def wendland_3_1(x):
    return (1 + (4 * x)) * ((1 - x) ** 4)
