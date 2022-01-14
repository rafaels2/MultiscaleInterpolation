from Config.Options import options

register_manifold = options.get_type_register("manifold")

from . import Circle
from . import RealNumbers
from . import RigidRotations
from . import SymmetricPositiveDefinite
from . import Euclidean
