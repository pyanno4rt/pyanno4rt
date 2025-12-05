"""
Data preprocessing module.

==================================================================

The module aims to provide methods and classes for data preprocessing, i.e., \
data cleaning, reduction, (re-)sampling and scaling.
"""

# Author: Tim Ortkamp

from . import scalers

from ._data_preprocessor import DataPreprocessor

__all__ = [
    'scalers',
    'DataPreprocessor']
