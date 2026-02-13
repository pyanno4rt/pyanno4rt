"""
Custom solvers module.

==================================================================

This module aims to provide methods and classes for custom solvers.
"""

# Author: Tim Ortkamp

from ._cmaes import CMAES
from ._lrcmaes import LRCMAES

__all__ = [
    'CMAES',
    'LRCMAES']
