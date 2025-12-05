"""
Learning module.

==================================================================

The module aims to provide methods and classes for data handling, \
preprocessing, model fitting, inspection & evaluation.
"""

# Author: Tim Ortkamp

from . import (
    dataset, evaluation, features, inspection, losses, preprocessing,
    tune_spaces)

from ._data_model_handler import DataModelHandler

__all__ = [
    'dataset',
    'evaluation',
    'features',
    'inspection',
    'losses',
    'preprocessing',
    'tune_spaces',
    'DataModelHandler']
