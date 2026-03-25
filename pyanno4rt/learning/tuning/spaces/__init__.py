"""
Tune spaces module.

==================================================================

The module aims to provide methods and classes for setting up the search \
spaces for the Bayesian/randomized hyperparameter tuner.
"""

# Author: Tim Ortkamp

from pyanno4rt.learning.tuning.spaces._tune_space_dt import TuneSpaceDT
from pyanno4rt.learning.tuning.spaces._tune_space_knn import TuneSpaceKNN
from pyanno4rt.learning.tuning.spaces._tune_space_lr import TuneSpaceLR
from pyanno4rt.learning.tuning.spaces._tune_space_nb import TuneSpaceNB
from pyanno4rt.learning.tuning.spaces._tune_space_nn import TuneSpaceNN
from pyanno4rt.learning.tuning.spaces._tune_space_rf import TuneSpaceRF
from pyanno4rt.learning.tuning.spaces._tune_space_soft_dt import TuneSpaceSoftDT
from pyanno4rt.learning.tuning.spaces._tune_space_svm import TuneSpaceSVM

__all__ = [
    'TuneSpaceDT',
    'TuneSpaceKNN',
    'TuneSpaceLR',
    'TuneSpaceNB',
    'TuneSpaceNN',
    'TuneSpaceRF',
    'TuneSpaceSoftDT',
    'TuneSpaceSVM']
