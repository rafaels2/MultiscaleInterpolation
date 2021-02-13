from ApproximationMethods.Quasi import Quasi


class MovingLeastSquares(Quasi):
    @staticmethod
    def _get_weights_for_point(point, x, y):
        return point.phi(x, y) * point.lambdas(x, y)

    @staticmethod
    def _normalize_weights(weights):
        return weights
