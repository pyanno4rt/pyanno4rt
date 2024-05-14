"""
Model inspection module.

==================================================================

The module aims to provide methods and classes to inspect the machine \
learning outcome models.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._model_inspector import ModelInspector

from . import algorithms

__all__ = ['ModelInspector',
           'algorithms']
