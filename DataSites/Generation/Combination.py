"""
This module allows to generate data sites with different configurations and types.
Not tested
"""
from Config.Options import options


class SamplingPointsCollection(object):
    def __init__(self, rbf_radius, function_to_evaluate, grids_parameters, **kwargs):
        # TODO: This should be a generation method
        self._grids = [
            options.get_option("generation_method", name)(
                rbf_radius, function_to_evaluate, parameters, **kwargs
            )
            for name, parameters in grids_parameters
        ]
