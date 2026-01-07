"""Kullback-Leibler divergence loss."""

# Author: Tim Ortkamp

# %% External package import

from numpy import clip, log, mean

# %% Function definition


def kl_divergence_loss(true_labels, predicted_labels):
    """
    Compute the Kullback-Leibler divergence loss.

    Parameters
    ----------
    true_labels : ndarray
        Ground truth label values.

    predicted_labels : ndarray
        Predicted label values.

    Returns
    -------
    float
        Kullback-Leibler divergence loss value.
    """

    # Clip the predictions
    epsilon = 1e-10
    true_labels = clip(true_labels, epsilon, 1-epsilon)
    predicted_labels = clip(predicted_labels, epsilon, 1-epsilon)

    # Calculate the KL divergence loss terms
    positive = true_labels*log(true_labels/predicted_labels)
    negative = (1-true_labels)*log((1-true_labels)/(1-predicted_labels))

    return mean(positive + negative)
