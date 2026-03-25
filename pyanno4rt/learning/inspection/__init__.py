"""
Inspection module.

==================================================================

The module aims to provide methods and classes for model inspection.
"""

# Author: Tim Ortkamp

from pyanno4rt.learning.inspection._feature_sensitivities import feature_sensitivities
from pyanno4rt.learning.inspection._permutation_importances import permutation_importances

from pyanno4rt.learning.inspection._model_inspector import ModelInspector

__all__ = [
    'feature_sensitivities',
    'permutation_importances',
    'ModelInspector']
