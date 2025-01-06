"""Neural network maps."""

# Author: Tim Ortkamp

# %% External package import

from tensorflow.keras.losses import (
    BinaryCrossentropy, BinaryFocalCrossentropy, KLDivergence)
from tensorflow.keras.optimizers import Adam, Ftrl, SGD

# %% Map definitions


loss_map = {'BCE': BinaryCrossentropy,
            'FocalBCE': BinaryFocalCrossentropy,
            'KLD': KLDivergence}

optimizer_map = {'Adam': Adam,
                 'Ftrl': Ftrl,
                 'SGD': SGD}
