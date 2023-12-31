"""Neural network losses."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from tensorflow.compat.v2.keras.losses import (
    BinaryCrossentropy, BinaryFocalCrossentropy, KLDivergence)

# %% Loss map


loss_map = {'BCE': BinaryCrossentropy,
            'FocalBCE': BinaryFocalCrossentropy,
            'KLD': KLDivergence}
