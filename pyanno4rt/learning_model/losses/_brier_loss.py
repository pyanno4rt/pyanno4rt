"""Brier score loss computation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from sklearn.metrics import brier_score_loss

# %% Function definition


def brier_loss(true_labels, predicted_labels):
    """
    Compute the Brier score loss.

    Parameters
    ----------
    true_labels : ndarray
        Ground truth label values.

    predicted_labels : ndarray
        Predicted label values.

    Returns
    -------
    float
        Brier score loss value.
    """

    return -brier_score_loss(true_labels, predicted_labels)
