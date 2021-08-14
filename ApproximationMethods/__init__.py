from Config.Options import options

register_approximation_method = options.get_type_register("approximation_method")

from . import AdaptiveQuasi
from . import MovingLeastSquares
from . import Naive
from . import NoNormalization
from . import Quasi
