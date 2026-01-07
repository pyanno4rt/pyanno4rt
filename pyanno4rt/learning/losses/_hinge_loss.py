"""Hinge loss."""

# Author: Tim Ortkamp

# %% External package import

from numpy import maximum, mean

# %% Function definition


def hinge_loss(true_labels, predicted_labels):
    """
    Compute the Hinge loss.

    Parameters
    ----------
    true_labels : ndarray
        Ground truth label values.

    predicted_labels : ndarray
        Predicted label values.

    Returns
    -------
    float
        Hinge loss value.
    """

    # Map the true labels to [-1, 1]
    scaled_true_labels = 2*true_labels-1

    return mean(maximum(0, 1-scaled_true_labels*predicted_labels))
