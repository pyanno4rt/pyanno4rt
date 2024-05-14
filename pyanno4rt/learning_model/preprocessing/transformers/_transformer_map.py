"""Transformers map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.learning_model.preprocessing.transformers import (
    Identity, StandardScaler, Whitening)

# %% Map definition


transformer_map = {'Identity': Identity,
                   'StandardScaler': StandardScaler,
                   'Whitening': Whitening}
