"""
Model evaluation module.

==================================================================

The module aims to provide methods and classes to evaluate the applied \
learning models.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._model_evaluator import ModelEvaluator

from . import metrics

__all__ = ['ModelEvaluator',
           'metrics']
