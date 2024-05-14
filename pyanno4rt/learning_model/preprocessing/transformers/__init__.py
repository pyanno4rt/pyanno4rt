"""
Data transformers module.

==================================================================

The module aims to provide methods and classes for data transformation.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._identity import Identity
from ._standard_scaler import StandardScaler
from ._whitening import Whitening

from ._transformer_map import transformer_map

__all__ = ['Identity',
           'StandardScaler',
           'Whitening',
           'transformer_map']
