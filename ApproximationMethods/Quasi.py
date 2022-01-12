"""
This is the main method we discuss.
Q(f)(x) = sum f(x_i) a(x) / sum a(x).
"""
from cachetools import cached

from Config.Config import config
from Config.Options import options
from Tools.Utils import generate_kernel, generate_cache
from .ApproximationMethod import ApproximationMethod
from . import register_approximation_method


@register_approximation_method("quasi")
class Quasi(ApproximationMethod):
    def __init__(
        self,
        original_function,
        grid_parameters,
        scale,
        manifold=None,
    ):
        """
        See the description of this file.
        :param original_function: f(x,y) -> manifold element
        :param grid_parameters: (x_min, x_max, y_min, y_max, fill_distance)
        :param scale: The rbf support radius.
        """
        if manifold is None:
            manifold = config.MANIFOLD

        super().__init__(
            manifold,
            original_function,
            grid_parameters,
            options.get_option("rbf", config.RBF),
        )
        self._is_approximating_on_tangent = config.IS_APPROXIMATING_ON_TANGENT
        self._rbf_radius = scale

        self._raw_data_sites = options.get_option(
            "generation_method", config.DATA_SITES_GENERATION
        )(*grid_parameters)

        self._data_sites = options.get_option(
            "data_storage", config.DATA_SITES_STORAGE
        )(
            self._raw_data_sites,
            self._rbf_radius,
            original_function,
            grid_parameters.fill_distance,
            phi_generator=self._calculate_phi,
        )

        self._kernel = generate_kernel(self._rbf, self._rbf_radius)

    @staticmethod
    def _get_weights_for_point(point, x, y):
        return point.phi(x, y)

    def _get_values_to_average(self, x, y):
        values_to_average = list()
        weights = list()

        for point in self._data_sites.points_in_radius(x, y):
            values_to_average.append(point.evaluation)
            weights.append(self._get_weights_for_point(point, x, y))

        return values_to_average, weights

    @staticmethod
    def _normalize_weights(weights):
        normalizer = sum(weights)

        if normalizer == 0:
            normalizer = 0.00001

        return [w_i / normalizer for w_i in weights]

    @cached(cache=generate_cache(maxsize=10000))
    def approximation(self, x, y):
        # TODO: point should be an array - not x, y. so we can generalize dimensions
        """ Average sampled points around (x, y), using phis as weights """
        values_to_average, weights = self._get_values_to_average(x, y)
        weights = self._normalize_weights(weights)
        # print(len(weights))

        if self._is_approximating_on_tangent:
            return sum(w_i * x_i for w_i, x_i in zip(weights, values_to_average))

        return self._manifold.average(values_to_average, weights)
