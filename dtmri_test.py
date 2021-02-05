import numpy as np

from DataSetParser import NRRDParser
from InputInterface import DTMRIDataSet
from Tools.SamplingPoints import SubDomain, GridParameters
from Tools.Visualization import BrainVisualizer


def _slice(func, index, offset=(0, 0)):
    def new_func(x, y):
        return func(np.array([x + offset[0], y + offset[1], index]))

    return new_func


def get_orig_confidence_and_size(slice_index):
    raw_data = NRRDParser('input_data\\gk2-rcc-mask.nhdr').parse()
    original = DTMRIDataSet(raw_data, np.array([0, 0, 0]), np.array([1, 1, 1]))
    confidence = original.confidence
    offset = (24, 50)
    trim_from_end = (60, 40)
    # raw_data.shape[1] - 1 - offset[0] - trim_from_end[0], raw_data.shape[2] - 1 - offset[1] - trim_from_end[1])
    return _slice(original, slice_index, offset), _slice(confidence, slice_index, offset), (
        110, 75)


def main():
    orig, confidence, size = get_orig_confidence_and_size(105)
    grid_params = GridParameters(0, size[0] - 1, 0, size[1] - 1, 1)
    grid = SubDomain(confidence, 3, orig, grid_params)
    evaluation, centers = grid.evaluation
    vis = BrainVisualizer(evaluation, centers)
    vis.show()


if __name__ == "__main__":
    main()
