from . import defaults


class Config(object):
    def __init__(self, base_config=None):
        self._base_config = base_config
        for setting in dir(defaults):
            if setting.isupper():
                setattr(self, setting, getattr(defaults, setting))

        if base_config is not None:
            self.update_config_with_diff(base_config)

    def update_config_with_diff(self, diff):
        for setting, value in diff.items():
            setattr(self, setting, value)

    def renew(self):
        self.__init__(base_config=self._base_config)

    def set_base_config(self, base_config):
        self._base_config = base_config

    def __repr__(self):
        representation = dict()
        for d in dir(self):
            if d.isupper():
                representation[d] = self.__getattribute__(d)

        return str(representation)


config = Config()
