"""
Tuning module.

==================================================================

The module aims to provide methods and classes for tuning the outcome model \
hyperparameters.
"""

# Author: Tim Ortkamp

from ._bayes_hp_tuner import BayesHPTuner
from ._grid_hp_tuner import GridHPTuner
from ._randomized_hp_tuner import RandomizedHPTuner

from .grids import (
    TuneGridDT, TuneGridKNN, TuneGridLR, TuneGridNB, TuneGridNN, TuneGridRF,
    TuneGridSVM)
from .spaces import (
    TuneSpaceDT, TuneSpaceKNN, TuneSpaceLR, TuneSpaceNB, TuneSpaceNN,
    TuneSpaceRF, TuneSpaceSVM)

__all__ = [
    'BayesHPTuner',
    'GridHPTuner',
    'RandomizedHPTuner',
    'TuneGridDT',
    'TuneGridKNN',
    'TuneGridLR',
    'TuneGridNB',
    'TuneGridNN',
    'TuneGridRF',
    'TuneGridSVM',
    'TuneSpaceDT',
    'TuneSpaceKNN',
    'TuneSpaceLR',
    'TuneSpaceNB',
    'TuneSpaceNN',
    'TuneSpaceRF',
    'TuneSpaceSVM']
