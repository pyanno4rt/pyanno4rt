"""
Input checking module.

==================================================================

This module aims to provide classes and functions to perform input parameter \
checks.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from . import check_functions
from . import check_maps

from ._input_checker import InputChecker

__all__ = ['check_functions',
           'check_maps',
           'InputChecker']
