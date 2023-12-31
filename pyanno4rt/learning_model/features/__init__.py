"""
Features module.

==================================================================

The module aims to provide methods and classes to handle the features of the \
base data set, i.e., mapping features to segments and definitions from the \
feature catalogue and iteratively (re)calculate the values as input to the \
learning model. In addition, the module contains the feature catalogue.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._feature_map_generator import FeatureMapGenerator
from ._feature_calculator import FeatureCalculator

from . import catalogue

__all__ = ['FeatureMapGenerator',
           'FeatureCalculator',
           'catalogue']
