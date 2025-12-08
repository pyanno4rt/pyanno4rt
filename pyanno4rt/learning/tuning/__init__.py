"""
Tuning module.

==================================================================

The module aims to provide methods and classes for tuning the outcome model \
hyperparameters.
"""

# Author: Tim Ortkamp

from ._bayes_hp_tuner import BayesHPTuner
from ._grid_hp_tuner import GridHPTuner
from ._random_HP_tuner import RandomHPTuner

from .spaces import (
    TuneSpaceDT, TuneSpaceKNN, TuneSpaceLR, TuneSpaceNB, TuneSpaceNN,
    TuneSpaceRF, TuneSpaceSVM)

__all__ = [
    'BayesHPTuner',
    'GridHPTuner',
    'RandomHPTuner',
    'TuneSpaceDT',
    'TuneSpaceKNN',
    'TuneSpaceLR',
    'TuneSpaceNB',
    'TuneSpaceNN',
    'TuneSpaceRF',
    'TuneSpaceSVM']
