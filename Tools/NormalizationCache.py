import json5
import os

import numpy as np

NORMALIZER_FILE = "C:\\temp\\normalizers.json5"


class NormalizationCache(object):

    def __init__(self, filename):
        self._filename = filename
        self._items_on_trial = set()

        if os.path.exists(filename):
            with open(filename, "r") as f:
                self._cache = json5.load(f)
        else:
            self._cache = {
                # ("3_1", 0.375, 0.75): 1/0.38762736,
                # ("3_1", 0.28125, 0.5625): 1/0.21791627,
                # ("3_1", 0.2109375, 0.421875): 1/0.11524012,
                # ("3_1", 0.158203125, 0.31640625): 1/0.06187601,
                # ("3_1", 0.11865234375, 0.2373046875): 1/0.033145268,
            }

    def update(self):
        with open(self._filename, "w") as f:
            json5.dump(self._cache, f)

    def __setitem__(self, key, value):
        self._cache[repr(key)] = str(value)
        self.update()

    def __getitem__(self, item):
        if repr(item) not in self._cache:
            # raise KeyError(f'Please calibrate first {item}')
            print(f"{item} is set to 1, will update...")
            return 1

        return np.float(self._cache[repr(item)])
