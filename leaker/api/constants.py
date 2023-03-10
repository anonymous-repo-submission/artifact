"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber

"""
import os
from enum import Enum

WHOOSH_INDEX_DIRECTORY: str = "data/whoosh/"

PICKLE_DIRECTORY: str = "data/pickle/"

RANGE_PICKLE_ID: str = "idx_values"

RANGE_QLOG_PICKLE_ID: str = "idx_queries"

FIGURE_DIRECTORY: str = "data/figures/"

WRITING_INTERVAL: int = 5000000

MIN_USER_QUERYLOG_ACTIVITY: int = 500

COMPILE_TIMEOUT: int = 1200  # in seconds

PYTHON_DIST_PACKAGES_DIRECTORY = "/usr/lib/python3/dist-packages/"

CACHE_DIRECTORY: str = "data/cache/"

if not os.path.exists(CACHE_DIRECTORY):
    os.makedirs(CACHE_DIRECTORY)


class AbortException(Exception):
    pass


class Selectivity(Enum):
    """
    Possible values for the selectivity of keywords.

    Independent - the query space is populated uniformly at random from the keywords in the data set
    High - the query space is populated with the highest selectivity keywords in the data set
    Low - the query space is populated with the lowest selectivity keywords in the data set
    PseudoLow - the query space is populated with the lowest selectivity keywords with a selectivity of at least 10
    """
    Independent = -1
    High = 0
    Low = 1
    PseudoLow = 2
