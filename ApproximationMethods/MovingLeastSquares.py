"""
This method promises polynomial reproduction, using the lambdas (of the reproduction).
"""
from ApproximationMethods.Quasi import Quasi
from . import register_approximation_method


@register_approximation_method("moving")
class MovingLeastSquares(Quasi):
    @staticmethod
    def _get_weights_for_point(point, x, y):
        return point.phi(x, y) * point.lambdas(x, y)

    @staticmethod
    def _normalize_weights(weights):
        return weights
