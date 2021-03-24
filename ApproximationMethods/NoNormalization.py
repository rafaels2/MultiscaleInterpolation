from ApproximationMethods.Quasi import Quasi
from Tools.NormalizationCache import NORMALIZER_FILE, NormalizationCache

normalization_cache = NormalizationCache(NORMALIZER_FILE)


class NoNormalization(Quasi):
    def __init__(self, *args):
        super(NoNormalization, self).__init__(*args)
        # self._normalizer = normalization_cache[(self._rbf.__name__, self._grid_parameters[0][1].mesh_norm,
        #                                         self._rbf_radius)]
        # self._normalizer = 1
        self._normalizer = 1.280153573980123

    def _normalize_weights(self, weights):
        return [w_i / self._normalizer for w_i in weights]
