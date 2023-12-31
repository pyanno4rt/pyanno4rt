"""Neural network optimizers."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from tensorflow.keras.optimizers import Adam, Ftrl, SGD

# %% Optimizers map


optimizer_map = {'Adam': Adam,
                 'Ftrl': Ftrl,
                 'SGD': SGD}
