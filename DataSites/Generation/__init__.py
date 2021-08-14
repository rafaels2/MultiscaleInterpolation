from Config.Options import options

register_generation = options.get_type_register("generation_method")

from . import Grid
from . import Halton
