"""Losses map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.learning_model.losses import brier_loss, log_loss

# %% Map definition


loss_map = {'Brier score': brier_loss,
            'Logloss': log_loss}
