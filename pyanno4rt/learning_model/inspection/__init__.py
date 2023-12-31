"""
Model inspection module.

==================================================================

The module aims to provide methods and classes to inspect the applied \
learning models.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._model_inspector import ModelInspector

from . import inspections

__all__ = ['ModelInspector',
           'inspections']
