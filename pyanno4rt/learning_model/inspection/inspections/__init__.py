"""
Inspection algorithms module.

==================================================================

The module aims to provide methods and classes to inspect the applied \
learning models.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._permutation_importance import PermutationImportance

__all__ = ['PermutationImportance']
