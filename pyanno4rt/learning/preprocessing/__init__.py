"""
Data preprocessing module.

==================================================================

The module aims to provide methods and classes for data preprocessing, i.e., \
data cleaning, reduction, (re-)sampling and transformation.
"""

# Author: Tim Ortkamp

# Import the preprocessing algorithms
from ._identity import Identity
from ._standard_scaler import StandardScaler
from ._whitening import Whitening

# Import the data preprocessor
from ._data_preprocessor import DataPreprocessor

__all__ = [
    'Identity',
    'StandardScaler',
    'Whitening',
    'DataPreprocessor']
