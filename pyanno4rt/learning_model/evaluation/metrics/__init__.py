"""
Evaluation metrics module.

==================================================================

The module aims to provide methods and classes to evaluate the applied \
learning models.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._f1_score import F1Score
from ._model_kpi import ModelKPI
from ._pr_score import PRScore
from ._roc_score import ROCScore

__all__ = ['F1Score',
           'ModelKPI',
           'PRScore',
           'ROCScore']
