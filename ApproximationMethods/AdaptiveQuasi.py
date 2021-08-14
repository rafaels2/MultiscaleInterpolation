from cachetools import cached

from ApproximationMethods.Quasi import Quasi
from Tools.Utils import generate_cache
from . import register_approximation_method


@register_approximation_method("adaptive_quasi")
class AdaptiveQuasi(Quasi):
    def __init__(self, original_function, grid_parameters, scale):
        if isinstance(original_function, tuple):
            original_function = combine(*original_function)
            self._is_adaptive = True
        else:
            self._is_adaptive = False
        super(AdaptiveQuasi, self).__init__(original_function, grid_parameters, scale)

    def _get_values_to_average(self, x, y):
        values_to_average = list()
        weights = list()

        base = self._original_function(x, y)[1]
        for point in self._data_sites.points_in_radius(x, y):
            values_to_average.append(self._manifold.exp(base, point.evaluation[0]))
            weights.append(self._get_weights_for_point(point, x, y))

        return values_to_average, weights

    @cached(cache=generate_cache(maxsize=1000))
    def approximation(self, x, y):
        base = self._original_function(x, y)[1]
        return self._manifold.log(base, super().approximation(x, y))
