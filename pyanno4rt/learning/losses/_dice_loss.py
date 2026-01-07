"""Dice loss."""

# Author: Tim Ortkamp

# %% External package import

from numpy import sum as nsum

# %% Function definition


def dice_loss(true_labels, predicted_labels, smoothing=1e-6):
    """
    Compute the Dice loss.

    Parameters
    ----------
    true_labels : ndarray
        Ground truth label values.

    predicted_labels : ndarray
        Predicted label values.

    smoothing : float, default=1e-6
        Smoothing parameter.

    Returns
    -------
    float
        Dice loss value.
    """

    # Flatten the label values
    true_labels = true_labels.flatten()
    predicted_labels = predicted_labels.flatten()

    # Calculate intersection and union
    intersection = nsum(true_labels*predicted_labels)
    union = nsum(true_labels) + nsum(predicted_labels)

    return 1-(2.0*intersection+smoothing)/(union+smoothing)
