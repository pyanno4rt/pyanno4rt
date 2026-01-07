"""
Losses module.

==================================================================

The module aims to provide functions to compute different learning model \
losses.
"""

# Author: Tim Ortkamp

from ._auc_loss import auc_loss
from ._brier_loss import brier_loss
from ._dice_loss import dice_loss
from ._focal_loss import focal_loss
from ._hinge_loss import hinge_loss
from ._kl_divergence_loss import kl_divergence_loss
from ._log_loss import log_loss

__all__ = [
    'auc_loss',
    'brier_loss',
    'dice_loss',
    'focal_loss',
    'hinge_loss',
    'kl_divergence_loss',
    'log_loss']
