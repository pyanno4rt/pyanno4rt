"""
Scalers module.

==================================================================

The module aims to provide methods and classes for data scaling.
"""

# Author: Tim Ortkamp

from ._identity import Identity
from ._standard_scaler import StandardScaler
from ._whitening import Whitening

__all__ = [
    'Identity',
    'StandardScaler',
    'Whitening']
