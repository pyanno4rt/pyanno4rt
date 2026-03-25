"""
Learning module.

==================================================================

The module aims to provide methods and classes for data handling, \
preprocessing, model fitting, inspection & evaluation.
"""

# Author: Tim Ortkamp

from pyanno4rt.learning import (
    datasets, evaluation, features, inspection, losses, models, preprocessing,
    tuning)

from pyanno4rt.learning._data_model_handler import DataModelHandler

__all__ = [
    'datasets',
    'evaluation',
    'features',
    'inspection',
    'losses',
    'models',
    'preprocessing',
    'tuning',
    'DataModelHandler']
