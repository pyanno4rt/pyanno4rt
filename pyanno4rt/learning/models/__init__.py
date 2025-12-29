"""
Models module.

==================================================================

The module aims to provide methods and classes for outcome modeling.
"""

# Author: Tim Ortkamp

from .logistic import LogisticRegression
from .naive_bayes import NaiveBayes
from .svm import SupportVectorMachine

__all__ = [
    'LogisticRegression',
    'NaiveBayes',
    'SupportVectorMachine']
