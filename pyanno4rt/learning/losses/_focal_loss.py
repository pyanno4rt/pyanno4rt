"""Focal loss."""

# Author: Tim Ortkamp

# %% External package import

from numpy import clip, log, mean, where

# %% Function definition


def focal_loss(true_labels, predicted_labels, gamma=2.0, alpha=0.25):
    """
    Compute the focal loss.

    Parameters
    ----------
    true_labels : ndarray
        Ground truth label values.

    predicted_labels : ndarray
        Predicted label values.

    gamma : float, default=2.0
        Focusing parameter.

    alpha : float, default=0.25
        Class balancing factor.

    Returns
    -------
    float
        Focal loss value.
    """

    # Clip the predicted labels
    epsilon = 1e-15
    predicted_labels = clip(predicted_labels, epsilon, 1-epsilon)

    # Calculate the binary crossentropy terms
    true_probability = where(
        true_labels == 1, predicted_labels, 1-predicted_labels)

    # Calculate the balancing weights
    alpha_weight = where(true_labels == 1, alpha, 1-alpha)

    # Calculate the modulation factor
    mod_factor = (1.0-true_probability)**gamma

    return mean(-alpha_weight*mod_factor*log(true_probability))
