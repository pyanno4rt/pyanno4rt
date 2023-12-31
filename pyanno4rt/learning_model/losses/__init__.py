"""
Losses module.

==================================================================

The module aims to provide loss functions to support the model training.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._brier_loss import brier_loss
from ._log_loss import log_loss

__all__ = ['brier_loss',
           'log_loss']
