"""
Cleaning module.

==================================================================

The module aims to provide methods and classes for data cleaning.
"""

# Author: Tim Ortkamp

from pyanno4rt.learning.preprocessing.cleaning._isolation_forest import IsolationForest
from pyanno4rt.learning.preprocessing.cleaning._local_outlier_factor import LocalOutlierFactor
from pyanno4rt.learning.preprocessing.cleaning._minimum_covariance_determinant import MinimumCovarianceDeterminant

__all__ = [
    'IsolationForest',
    'LocalOutlierFactor',
    'MinimumCovarianceDeterminant']
