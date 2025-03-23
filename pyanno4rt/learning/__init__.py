"""
Learning module.

==================================================================

The module aims to provide methods and classes for data handling, \
preprocessing, learning model fitting, inspection & evaluation.
"""

# Author: Tim Ortkamp

# Import the main classes
from ._data_model_handler import DataModelHandler
from ._machine_learning_model import MachineLearningModel

# Import the submodules
from . import dataset
from . import evaluation
from . import features
from . import forest
from . import inspection
from . import logistic
from . import losses
from . import naive_bayes
from . import neighbors
from . import neural_network
from . import preprocessing
from . import svm
from . import tree

__all__ = [
    'DataModelHandler',
    'MachineLearningModel',
    'dataset',
    'evaluation',
    'features',
    'forest',
    'inspection',
    'logistic',
    'losses',
    'naive_bayes',
    'neighbors',
    'neural_network',
    'preprocessing',
    'svm',
    'tree']
