class Options:
    def __init__(self):
        self._types = dict()

    def new_type(self, name):
        self._types[name] = dict()

    def get_options(self, type_name):
        return self._types[type_name]

    def add_option(self, type_name, option_name, value):
        self._types[type_name][option_name] = value

    def get_option(self, type_name, option_name):
        return self._types[type_name][option_name]

    def get_type_register(self, type_name):
        self.new_type(type_name)

        def register(option_name):
            def decorator(obj):
                self.add_option(type_name, option_name, obj)
                return obj

            return decorator

        return register


options = Options()

import Manifolds
import ApproximationMethods
import OriginalFunction
import RBF
import DataSites.Generation
import DataSites.Storage
