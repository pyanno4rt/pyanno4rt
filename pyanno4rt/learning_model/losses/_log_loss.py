"""Log loss computation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from tensorflow import cast, float64
from tensorflow.keras.losses import binary_crossentropy

# %% Function definition


def log_loss(true_labels, predicted_labels):
    """
    Compute the log loss.

    Parameters
    ----------
    true_labels : ndarray
        Ground truth label values.

    predicted_labels : ndarray
        Predicted label values.

    Returns
    -------
    float
        Log loss value.
    """

    return -binary_crossentropy(
        cast(true_labels, float64),
        cast(predicted_labels, float64)).numpy().mean()
