"""
Learning module.

==================================================================

The module aims to provide methods and classes for data handling, \
preprocessing, model fitting, inspection & evaluation.
"""

# Author: Tim Ortkamp

from . import (
    datasets, evaluation, features, inspection, models, preprocessing, tuning)

from ._data_model_handler import DataModelHandler

__all__ = [
    'datasets',
    'evaluation',
    'features',
    'inspection',
    'models',
    'preprocessing',
    'tuning',
    'DataModelHandler']
