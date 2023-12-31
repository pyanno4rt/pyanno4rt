"""
Input checking module.

==================================================================

The module aims to provide classes and functions to perform input parameter \
checks and guarantee the program integrity.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._input_checker import InputChecker

from . import check_functions
from . import check_maps

__all__ = ['InputChecker',
           'check_functions',
           'check_maps']
