from cachetools import cached

from ApproximationMethods.Quasi import Quasi
from Tools.Utils import generate_cache


class AdaptiveQuasi(Quasi):
    def _get_values_to_average(self, x, y):
        values_to_average = list()
        weights = list()

        base = self._original_function(x, y)[1]
        for point in self._grid.points_in_radius(x, y):
            values_to_average.append(self._manifold.exp(base, point.evaluation[0]))
            weights.append(self._get_weights_for_point(point, x, y))

        return values_to_average, weights

    @cached(cache=generate_cache(maxsize=1000))
    def approximation(self, x, y):
        base = self._original_function(x, y)[1]
        return self._manifold.log(base, super().approximation(x, y))
