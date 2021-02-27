from ApproximationMethods.Quasi import Quasi
import json5
import os

NORMALIZER_FILE = "C:\\temp\\normalizers.json5"


class NormalizationCache(object):

    def __init__(self, filename):
        self._filename = filename
        self._items_on_trial = set()

        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self._cache = json5.load(f)
        else:
            self._cache = {
                (0.375, 0.75): 1/0.38762736,
                (0.28125, 0.5625): 1/0.21791627,
                (0.2109375, 0.421875): 1/0.11524012,
                (0.158203125, 0.31640625): 1/0.06187601,
                (0.11865234375, 0.2373046875): 1/0.033145268,
            }

    def update(self):
        with open(self._filename, "wb") as f:
            json5.dump(self._cache, f)

    def __setitem__(self, key, value):
        if key in self._items_on_trial:
            self._cache[key] = value
            self._items_on_trial.remove(key)
            self.update()

    def __getitem__(self, item):
        if item not in self._cache:
            print(f"{item} is set to 1, will update...")
            self._items_on_trial.add(item)
            return 1

        return self._cache[item]


normalization_cache = NormalizationCache(NORMALIZER_FILE)


class NoNormalization(Quasi):
    def __init__(self, *args):
        super(NoNormalization, self).__init__(*args)
        self._normalizer = normalization_cache[(self._grid_parameters[0][1].mesh_norm, self._rbf_radius)]

    def _normalize_weights(self, weights):
        return [w_i / self._normalizer for w_i in weights]
