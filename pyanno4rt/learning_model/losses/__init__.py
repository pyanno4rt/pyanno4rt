"""
Losses module.

==================================================================

The module aims to provide functions to compute different machine learning \
outcome model losses.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._brier_loss import brier_loss
from ._log_loss import log_loss

from ._loss_map import loss_map

__all__ = ['brier_loss',
           'log_loss',
           'loss_map']
