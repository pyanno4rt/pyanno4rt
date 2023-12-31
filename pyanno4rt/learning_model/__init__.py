"""
Learning model module.

==================================================================

The module aims to provide methods and classes for data handling, \
preprocessing, learning model training, inspection & evaluation.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._data_model_handler import DataModelHandler

from . import data
from . import evaluation
from . import features
from . import frequentist
from . import inspection
from . import losses
from . import preprocessing

__all__ = ['DataModelHandler',
           'data',
           'evaluation',
           'features',
           'frequentist',
           'inspection',
           'losses',
           'preprocessing']
