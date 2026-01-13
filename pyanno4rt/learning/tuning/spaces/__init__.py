"""
Tune spaces module.

==================================================================

The module aims to provide methods and classes for setting up the search \
spaces for the Bayesian/randomized hyperparameter tuner.
"""

# Author: Tim Ortkamp

from ._tune_space_dt import TuneSpaceDT
from ._tune_space_knn import TuneSpaceKNN
from ._tune_space_lr import TuneSpaceLR
from ._tune_space_nb import TuneSpaceNB
from ._tune_space_nn import TuneSpaceNN
from ._tune_space_rf import TuneSpaceRF
from ._tune_space_svm import TuneSpaceSVM

__all__ = [
    'TuneSpaceDT',
    'TuneSpaceKNN',
    'TuneSpaceLR',
    'TuneSpaceNB',
    'TuneSpaceNN',
    'TuneSpaceRF',
    'TuneSpaceSVM']
