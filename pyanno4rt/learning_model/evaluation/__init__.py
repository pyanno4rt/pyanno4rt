"""
Model evaluation module.

==================================================================

The module aims to provide methods and classes to evaluate the machine \
learning outcome models.
"""

# Author: Tim Ortkamp

from ._model_evaluator import ModelEvaluator

from . import metrics

__all__ = [
    'ModelEvaluator',
    'metrics']
