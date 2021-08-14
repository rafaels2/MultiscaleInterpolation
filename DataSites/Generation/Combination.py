class SamplingPointsCollection(object):
    def __init__(self, rbf_radius, function_to_evaluate, grids_parameters, **kwargs):
        # TODO: This should be a generation method
        self._grids = [
            SAMPLING_POINTS_CLASSES[name](
                rbf_radius, function_to_evaluate, parameters, **kwargs
            )
            for name, parameters in grids_parameters
        ]
