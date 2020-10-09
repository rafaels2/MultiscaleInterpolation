from Tools.SamplingPoints import SubDomain, GridParameters
from Tools.Visualization import BrainVisualizer
from DataSetParser import NRRDParser
from InputInterface import DTMRIDataSet
import numpy as np


def _slice(func, index):
    def new_func(x, y):
        return func(np.array([x, y, index]))

    return new_func


def main():
    raw_data = NRRDParser('input_data\gk2-rcc-mask.nhdr').parse()
    # raw_data = NRRDParser('input_data\dt-helix.nhdr').parse()
    slice_index = 105
    original = DTMRIDataSet(raw_data, np.array([0, 0, 0]), np.array([1, 1, 1]))
    confidence = original.confidence
    grid_params = GridParameters(0, raw_data.shape[1] - 1, 0, raw_data.shape[2] - 1, 1)
    # def __init__(self, rbf_radius, function_to_evaluate, grid_parameters, phi_generator=None, *args, **kwargs):
    grid = SubDomain(_slice(confidence, slice_index), 3, _slice(original, slice_index), grid_params)
    evaluation, centers = grid.evaluation
    vis = BrainVisualizer(evaluation, centers, max(raw_data.shape[1], raw_data.shape[2]), 1)
    vis.show()


if __name__ == "__main__":
    main()
