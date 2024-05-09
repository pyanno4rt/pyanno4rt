"""
Treatment plan evaluation module.

==================================================================

This module aims to provide methods and classes to evaluate the generated \
treatment plans.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._dosimetrics_evaluator import DosimetricsEvaluator
from ._dvh_evaluator import DVHEvaluator

__all__ = ['DosimetricsEvaluator',
           'DVHEvaluator']
