"""
Inspection module.

==================================================================

The module aims to provide methods and classes for model inspection.
"""

# Author: Tim Ortkamp

from ._feature_sensitivities import feature_sensitivities
from ._permutation_importances import permutation_importances

from ._model_inspector import ModelInspector

__all__ = [
    'feature_sensitivities',
    'permutation_importances',
    'ModelInspector']
