"""Neural network maps."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from tensorflow.keras.losses import (
    BinaryCrossentropy, BinaryFocalCrossentropy, KLDivergence)
from tensorflow.keras.optimizers import Adam, Ftrl, SGD

# %% Maps


loss_map = {'BCE': BinaryCrossentropy,
            'FocalBCE': BinaryFocalCrossentropy,
            'KLD': KLDivergence}

optimizer_map = {'Adam': Adam,
                 'Ftrl': Ftrl,
                 'SGD': SGD}
