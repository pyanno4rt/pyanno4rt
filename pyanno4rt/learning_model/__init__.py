"""
Learning model module.

==================================================================

The module aims to provide methods and classes for data handling, \
preprocessing, learning model fitting, inspection & evaluation.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from . import dataset
from . import evaluation
from . import features
from . import frequentist
from . import inspection
from . import losses
from . import preprocessing

from ._data_model_handler import DataModelHandler

__all__ = ['dataset',
           'evaluation',
           'features',
           'frequentist',
           'inspection',
           'losses',
           'preprocessing',
           'DataModelHandler']
