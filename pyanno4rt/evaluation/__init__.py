"""
Treatment plan evaluation module.

==================================================================

The module aims to provide methods and classes to evaluate the generated \
treatment plans.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._dosimetrics import Dosimetrics
from ._dvh import DVH

__all__ = ['Dosimetrics',
           'DVH']
