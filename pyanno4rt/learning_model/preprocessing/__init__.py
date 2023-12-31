"""
Data preprocessing module.

==================================================================

The module aims to provide methods and classes for data preprocessing, i.e., \
for building up a flexible preprocessing pipeline with data cleaning, \
reduction, and transformation algorithms.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._data_preprocessor import DataPreprocessor

from . import cleaners
from . import reducers
from . import samplers
from . import transformers

__all__ = ['DataPreprocessor',
           'cleaners',
           'reducers',
           'samplers',
           'transformers']
