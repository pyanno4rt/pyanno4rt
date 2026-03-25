"""
Tuning module.

==================================================================

The module aims to provide methods and classes for tuning the outcome model \
hyperparameters.
"""

# Author: Tim Ortkamp

from pyanno4rt.learning.tuning._bayes_hp_tuner import BayesHPTuner
from pyanno4rt.learning.tuning._grid_hp_tuner import GridHPTuner
from pyanno4rt.learning.tuning._randomized_hp_tuner import RandomizedHPTuner

from pyanno4rt.learning.tuning.grids import (
    TuneGridDT, TuneGridKNN, TuneGridLR, TuneGridNB, TuneGridNN, TuneGridRF,
    TuneGridSoftDT, TuneGridSVM)
from pyanno4rt.learning.tuning.spaces import (
    TuneSpaceDT, TuneSpaceKNN, TuneSpaceLR, TuneSpaceNB, TuneSpaceNN,
    TuneSpaceRF, TuneSpaceSoftDT, TuneSpaceSVM)

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
    'TuneGridSoftDT',
    'TuneGridSVM',
    'TuneSpaceDT',
    'TuneSpaceKNN',
    'TuneSpaceLR',
    'TuneSpaceNB',
    'TuneSpaceNN',
    'TuneSpaceRF',
    'TuneSpaceSoftDT',
    'TuneSpaceSVM']
