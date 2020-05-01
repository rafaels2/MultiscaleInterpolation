MANIFOLDS = {}


def register_manifold(name):
	def _cls(cls):
		MANIFOLDS[name] = cls
		return cls
	return _cls


from . import Circle
from . import RealNumbers
from . import RigidRotations
from . import SymmetricPositiveDefinite
