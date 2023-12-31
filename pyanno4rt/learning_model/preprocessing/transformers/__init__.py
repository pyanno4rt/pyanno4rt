"""
Transformers module.

==================================================================

The module aims to provide methods and classes for data transformation in the \
context of data preprocessing.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._equalizer import Equalizer
from ._standard_scaler import StandardScaler
from ._whitening import Whitening

__all__ = ['Equalizer',
           'StandardScaler',
           'Whitening']
