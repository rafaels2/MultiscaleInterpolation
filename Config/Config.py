from . import defaults


class Config(object):
    """Configurations based on defaults module.
    Every object there is an attribute of the config file"""

    def __init__(self, base_config=None):
        self._base_config = base_config
        for setting in dir(defaults):
            if setting.isupper():
                setattr(self, setting, getattr(defaults, setting))

        if base_config is not None:
            self.update_config_with_diff(base_config)
        self.scale_index = 0

    def update_config_with_diff(self, diff):
        """Update current config with diff dict"""
        for setting, value in diff.items():
            setattr(self, setting, value)

    def renew(self):
        """initialize the configurations to base config"""
        self.__init__(base_config=self._base_config)

    def set_base_config(self, base_config):
        """Update base config before renew"""
        self._base_config = base_config

    def __repr__(self):
        """This is here to print and debug"""
        representation = dict()
        for d in dir(self):
            if d.isupper():
                representation[d] = self.__getattribute__(d)

        return str(representation)


# Use this object to get configurations
config = Config()
