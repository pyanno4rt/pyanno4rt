"""
Cleaning module.

==================================================================

The module aims to provide methods and classes for data cleaning.
"""

# Author: Tim Ortkamp

from ._isolation_forest import IsolationForest
from ._local_outlier_factor import LocalOutlierFactor
from ._minimum_covariance_determinant import MinimumCovarianceDeterminant

__all__ = [
    'IsolationForest',
    'LocalOutlierFactor',
    'MinimumCovarianceDeterminant']
