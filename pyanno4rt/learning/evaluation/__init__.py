"""
Evaluation module.

==================================================================

The module aims to provide methods and classes to evaluate outcome models.
"""

# Author: Tim Ortkamp

from pyanno4rt.learning.evaluation._auc_pr import auc_pr
from pyanno4rt.learning.evaluation._auc_roc import auc_roc
from pyanno4rt.learning.evaluation._f1 import f1
from pyanno4rt.learning.evaluation._kpi import kpi

from pyanno4rt.learning.evaluation._model_evaluator import ModelEvaluator

__all__ = [
    'auc_pr',
    'auc_roc',
    'f1',
    'kpi',
    'ModelEvaluator']
