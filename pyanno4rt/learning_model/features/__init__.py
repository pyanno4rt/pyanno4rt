"""
Features module.

==================================================================

The module aims to provide methods and classes to handle the features of the \
base data set, i.e., feature map generation and iterative recalculation of \
the learning model inputs using the feature catalogue.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._feature_calculator import FeatureCalculator
from ._feature_map_generator import FeatureMapGenerator

from . import catalogue

__all__ = ['FeatureCalculator',
           'FeatureMapGenerator',
           'catalogue']
