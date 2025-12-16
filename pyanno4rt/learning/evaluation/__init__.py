"""
Evaluation module.

==================================================================

The module aims to provide methods and classes to evaluate outcome models.
"""

# Author: Tim Ortkamp

from ._auc_pr import auc_pr
from ._auc_roc import auc_roc
from ._f1 import f1
from ._kpi import kpi

from ._model_evaluator import ModelEvaluator

__all__ = [
    'auc_pr',
    'auc_roc',
    'f1',
    'kpi',
    'ModelEvaluator']
