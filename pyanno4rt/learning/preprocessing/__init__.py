"""
Preprocessing module.

==================================================================

The module aims to provide methods and classes for data preprocessing, i.e., \
data cleaning, reduction, (re-)sampling and scaling.
"""

# Author: Tim Ortkamp

from ._tabular_preprocessor import TabularPreprocessor

from .scalers import StandardScaler, Whitening

__all__ = [
    'TabularPreprocessor',
    'StandardScaler',
    'Whitening']
