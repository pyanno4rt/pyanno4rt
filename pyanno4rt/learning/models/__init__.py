"""
Models module.

==================================================================

The module aims to provide methods and classes for outcome modeling.
"""

# Author: Tim Ortkamp

from .logistic import LogisticRegression
from .svm import SupportVectorMachine

__all__ = [
    'LogisticRegression',
    'SupportVectorMachine']
