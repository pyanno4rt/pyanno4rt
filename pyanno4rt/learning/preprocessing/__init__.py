"""
Preprocessing module.

==================================================================

The module aims to provide methods and classes for data preprocessing, i.e., \
data cleaning, reduction, (re-)sampling and scaling.
"""

# Author: Tim Ortkamp

from ._tabular_preprocessor import TabularPreprocessor

from .cleaning import (
    IsolationForest, LocalOutlierFactor, MinimumCovarianceDeterminant)
from .reduction import PrincipalComponentAnalysis
from .transformation import StandardScaler, Whitening

__all__ = [
    'TabularPreprocessor',
    'IsolationForest',
    'LocalOutlierFactor',
    'MinimumCovarianceDeterminant',
    'PrincipalComponentAnalysis',
    'StandardScaler',
    'Whitening']
