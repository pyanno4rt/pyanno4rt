"""
Evaluation metrics module.

==================================================================

The module aims to provide functions to compute different evaluation metrics.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._auc_pr import auc_pr
from ._auc_roc import auc_roc
from ._f1 import f1
from ._kpi import kpi

__all__ = ['auc_pr',
           'auc_roc',
           'f1',
           'kpi']
