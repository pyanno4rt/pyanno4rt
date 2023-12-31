"""
Projections module.

==================================================================

The module aims to provide methods and classes to project the fluence onto \
the physical or the RBE-weighted dose and vice versa.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._backprojection import BackProjection
from ._constant_rbe_projection import ConstantRBEProjection
from ._dose_projection import DoseProjection

from ._projections_map import projections_map

__all__ = ['BackProjection',
           'ConstantRBEProjection',
           'DoseProjection',
           'projections_map']
