"""
Preprocessing module.

==================================================================

The module aims to provide methods and classes for data preprocessing, i.e., \
data cleaning, reduction, (re-)sampling and scaling.
"""

# Author: Tim Ortkamp

from pyanno4rt.learning.preprocessing._tabular_preprocessor import TabularPreprocessor

from pyanno4rt.learning.preprocessing.cleaning import (
    IsolationForest, LocalOutlierFactor, MinimumCovarianceDeterminant)
from pyanno4rt.learning.preprocessing.reduction import PrincipalComponentAnalysis
from pyanno4rt.learning.preprocessing.transformation import StandardScaler, Whitening

__all__ = [
    'TabularPreprocessor',
    'IsolationForest',
    'LocalOutlierFactor',
    'MinimumCovarianceDeterminant',
    'PrincipalComponentAnalysis',
    'StandardScaler',
    'Whitening']
