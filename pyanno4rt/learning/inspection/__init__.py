"""
Model inspection module.

==================================================================

The module aims to provide methods and classes to inspect the learning models.
"""

# Author: Tim Ortkamp

from ._permutation_importances import permutation_importances
from ._model_inspector import ModelInspector

__all__ = [
    'permutation_importances',
    'ModelInspector']
