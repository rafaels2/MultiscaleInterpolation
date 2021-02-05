from collections import namedtuple
from abc import abstractmethod
import numpy as np

from cachetools import cached, LFUCache

INDEX_TOLERANCE = 0.05
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DTMRIVoxel = namedtuple('DTMRIVoxel', 'confidence, xx, xy, xz, yy, yz, zz')
CACHE_SIZE = 100000


class AbstractInputInterface(object):
    @abstractmethod
    def __call__(self, p):
        """
        Each input object should return the original value for each point.
        :param p: The point, preferably np.ndarray
        :return: value
        """
        pass


class OriginalFunction(AbstractInputInterface):
    def __init__(self, func):
        self._func = func

    def __call__(self, p):
        return self._func(p)


class InputDataSet(AbstractInputInterface):
    def __init__(self, data_set, grid_offset, resolution):
        """
        This is the generic class to represent outer data sets.
        :param data_set: A multidimensional array representing the dataset.
        :param grid_offset: The position of the index [0,...,0].
        :param resolution: The distance between pixel in each axis. (Array sized by len(data_set.shape)).
        """
        self._data_set = data_set
        self._grid_offset = grid_offset
        self._resolution = resolution

    def _get_index_from_point(self, p):
        relative_location = p - self._grid_offset
        multi_index = np.zeros_like(self._grid_offset, dtype=int)

        for axis in np.ndindex(self._grid_offset.shape):
            index = relative_location[axis] * self._resolution[axis]
            multi_index[axis] = int(np.round(index))
            assert abs(multi_index[axis] - index) < INDEX_TOLERANCE, \
                f"no existing value for {p}, multi_index: {multi_index[axis]}, index: {index}"
            assert multi_index[axis] >= 0, \
                f"The point {p} is out of range"

        return multi_index

    def __call__(self, p):
        multi_index = self._get_index_from_point(p)
        return self._data_set[tuple(multi_index)]


class ConfidenceError(KeyError):
    # TODO: create a meaningful error message.
    pass


class DTMRIDecoder(object):
    # TODO: in order to deal with Confidence error, we should have a "find nearest" function.
    def __init__(self, data, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
        self._data = data
        self._confidence_threshold = confidence_threshold

    @cached(cache=LFUCache(maxsize=100000))
    def __getitem__(self, index):
        index = list(index)
        item = DTMRIVoxel(*self._data[tuple([slice(0, 7)] + index)])
        if item.confidence < self._confidence_threshold:
            raise ConfidenceError(f'item is: p{item}')

        return np.array([[item.xx, item.xy, item. xz],
                         [item.xy, item.yy, item.yz],
                         [item.xz, item.yz, item.zz]])


class DTMRIDataSet(InputDataSet):
    def __init__(self, data_set, grid_offset, resolution,
                 confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
        self._raw_data_set = data_set
        data_set = DTMRIDecoder(data_set, confidence_threshold)
        super().__init__(data_set, grid_offset, resolution)

    def confidence(self, p):
        multi_index = self._get_index_from_point(p)
        # TODO: debug this
        index = tuple([0] + list(multi_index))
        return self._raw_data_set[index]

# TODO: write a unittest here.
