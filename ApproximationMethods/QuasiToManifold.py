from cachetools import cached

from Config.Config import config
from Config.Options import options
from Tools.Utils import generate_cache
from . import register_approximation_method
from .ApproximationMethod import ApproximationMethod


@register_approximation_method("projection")
class QuasiToManifold(ApproximationMethod):
    def __init__(
        self,
        original_function,
        grid_parameters,
        scale,
    ):
        self._secondary_manifold = config.SECONDARY_MANIFOLD

        @cached(cache=generate_cache(maxsize=1000))
        def new_original_function(*args):
            return self._secondary_manifold.exp(
                self._secondary_manifold.zero_func(*args),
                self._secondary_manifold.from_euclid_to_tangent(
                    original_function(*args)
                ),
            )

        self._secondary_method = options.get_option(
            "approximation_method", config.SECONDARY_SCALED_INTERPOLATION_METHOD
        )(
            new_original_function,
            grid_parameters,
            scale,
            manifold=config.SECONDARY_MANIFOLD,
        )

    @cached(cache=generate_cache(maxsize=1000))
    def approximation(self, x, y):
        approximation = self._secondary_method.approximation(x, y)
        return self._secondary_manifold.log(
            self._secondary_manifold.zero_func(x, y), approximation
        )
