"""Log loss function."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from tensorflow.compat.v2 import cast, float64
from tensorflow.keras.losses import binary_crossentropy

# %% Function definition


def log_loss(true_labels, predicted_labels):
    """Compute the log loss from the true and predicted labels."""
    return -binary_crossentropy(
        cast(true_labels, float64),
        cast(predicted_labels, float64)).numpy().mean()
