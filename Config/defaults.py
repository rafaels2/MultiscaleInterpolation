"""
These are the default values for the configurations.
for example config.GRID_SIZE 's default value is 0.45
"""

# The size of the test grid. The default grid is symmetric [-GRID_SIZE, GIRD_SIZE]^2.
GRID_SIZE = 0.45

# A multiplier of the fill distance and rbf support
BASE_RESOLUTION = 2

# Number of scales in multiscale experiment
NUMBER_OF_SCALES = 3

# Fill distance on test grid
TEST_FILL_DISTANCE = 0.02

# Inner config of the runner.py - TODO remove
SCALING_FACTOR_POWER = 1

# Scaling factor - mu
SCALING_FACTOR = 0.75

# The label in the mses plot
MSE_LABEL = "Default Run"

# Option from DataSites.Generation
DATA_SITES_GENERATION = "grid"

# Option from DataSites.Storage
DATA_SITES_STORAGE = "kd-tree"

# Option from RBF
RBF = "wendland_3_1"

# Option from Manifolds
MANIFOLD = "numbers"
SECONDARY_MANIFOLD = "rotations"

EUCLIDEAN_DIMENSION = 3

# Option from ApproximationMethods
SCALED_INTERPOLATION_METHOD = "quasi"
SECONDARY_SCALED_INTERPOLATION_METHOD = "quasi"

# Option from OriginalFunction
ORIGINAL_FUNCTION = "numbers"

ERROR_CALC = True

BASE_SCALE = 1

CMAP = "viridis"
CB = True
CMAX = 0
IS_PROXIMITY = False
NOISE = "none"
DENOISE = False
DENOISE_THRESHOLD = 1
NOISE_SIGMA = 0.1

# The sampling addition to the test grid: [-GRID_SIZE - GRID_BORDER, GRID_SIZE + GRID_BORDER]
GRID_BORDER = 0.5

""" Not important for now """
OUTPUT_DIR = "results"
NAME = "temp"
EXECUTION_NAME = "ApproximationResults"
NORM_VISUALIZATION = False
IS_APPROXIMATING_ON_TANGENT = False
IS_ADAPTIVE = False
ERROR_CALC = False
