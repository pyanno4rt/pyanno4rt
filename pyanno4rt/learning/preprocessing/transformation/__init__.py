"""
Transformation module.

==================================================================

The module aims to provide methods and classes for data transformation.
"""

# Author: Tim Ortkamp

from ._standard_scaler import StandardScaler
from ._whitening import Whitening

__all__ = [
    'StandardScaler',
    'Whitening']
