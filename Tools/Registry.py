class Registry(object):
	def __init__():
		self._registry = {}

	def get_names():
		return self._registry.keys()

	def __item__(self, name):
		return self._registry[name]

	def register(self, name):
		def _cls(cls):
			self._registry[name] = cls
			return cls

		return _cls


MANIFOLDS = Registry()
MULTISCALE_METHODS = Registry()
