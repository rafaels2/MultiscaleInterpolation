class Options:
    """
    Registry of options - dict{str: obj}.
    """

    def __init__(self):
        self._types = dict()

    def new_type(self, name):
        """Add an options list"""
        self._types[name] = dict()

    def get_options(self, type_name):
        """Get all options in a list"""
        return self._types[type_name]

    def add_option(self, type_name, option_name, value):
        """Add an option to a list"""
        self._types[type_name][option_name] = value

    def get_option(self, type_name, option_name):
        """Get the option `option_name` from the list `type_name`"""
        return self._types[type_name][option_name]

    def get_type_register(self, type_name):
        """Add a new options list and get a register function"""
        self.new_type(type_name)

        def register(option_name):
            """The gets a decorator by name"""

            def decorator(obj):
                """
                Decorate with this decorator to list the
                obj in the `type_name` list under the name `option_name`
                """
                self.add_option(type_name, option_name, obj)
                return obj

            return decorator

        return register


# This is the options instance
options = Options()


import Manifolds
import ApproximationMethods
import OriginalFunction
import RBF
import DataSites.Generation
import DataSites.Storage
