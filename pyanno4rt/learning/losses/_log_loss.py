"""Log loss."""

# Author: Tim Ortkamp

# %% External package import

from sklearn.metrics import log_loss as sk_log_loss

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

    return sk_log_loss(true_labels, predicted_labels)
