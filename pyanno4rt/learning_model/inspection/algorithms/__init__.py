"""
Inspection algorithms module.

==================================================================

The module aims to provide functions to run different inspection algorithms.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._permutation_importances import permutation_importances

__all__ = ['permutation_importances']
