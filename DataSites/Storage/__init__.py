from Config.Options import options

add_sampling_class = options.get_type_register("data_storage")

from . import Grid
from . import KDTree
