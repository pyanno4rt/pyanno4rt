"""
Losses module.

==================================================================

The module aims to provide functions to compute different learning model \
losses.
"""

# Author: Tim Ortkamp

from pyanno4rt.learning.losses._auc_loss import auc_loss
from pyanno4rt.learning.losses._brier_loss import brier_loss
from pyanno4rt.learning.losses._dice_loss import dice_loss
from pyanno4rt.learning.losses._focal_loss import focal_loss
from pyanno4rt.learning.losses._hinge_loss import hinge_loss
from pyanno4rt.learning.losses._kl_divergence_loss import kl_divergence_loss
from pyanno4rt.learning.losses._log_loss import log_loss

__all__ = [
    'auc_loss',
    'brier_loss',
    'dice_loss',
    'focal_loss',
    'hinge_loss',
    'kl_divergence_loss',
    'log_loss']
