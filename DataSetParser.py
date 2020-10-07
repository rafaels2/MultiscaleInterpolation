from collections import namedtuple
from abc import abstractmethod
import numpy as np
import nrrd

DataSet = namedtuple('DataSet', 'data, resolution')


class AbstractDataSetParser(object):
    def __init__(self, path):
        self._path = path

    @abstractmethod
    def parse(self):
        """
        :return: DataSet
        """
        pass


class NRRDParser(AbstractDataSetParser):
    def parse(self):
        # TODO: add mins, maxs - it's good for grid initialization.
        reader, header = nrrd.read(self._path)
        min_axis = np.array(header['axis mins'][1:])
        max_axis = np.array(header['axis maxs'][1:])
        sizes = np.array(header['sizes'][1:])
        resolution = (max_axis - min_axis - np.ones_like(max_axis)) / sizes
        return DataSet(reader, resolution)
