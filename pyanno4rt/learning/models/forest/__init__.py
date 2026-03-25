"""
Random forest module.

==================================================================

The module aims to provide methods and classes for random forest outcome \
modeling.
"""

# Author: Tim Ortkamp

from pyanno4rt.learning.models.forest._projection_forest import ProjectionForest

from pyanno4rt.learning.models.forest._random_forest import RandomForest

__all__ = [
    'ProjectionForest',
    'RandomForest']
