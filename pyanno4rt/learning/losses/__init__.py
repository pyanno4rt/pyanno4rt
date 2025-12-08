"""
Losses module.

==================================================================

The module aims to provide functions to compute different learning model \
losses.
"""

# Author: Tim Ortkamp

from ._auc_loss import auc_loss
from ._brier_loss import brier_loss
from ._log_loss import log_loss

__all__ = [
    'auc_loss',
    'brier_loss',
    'log_loss']
