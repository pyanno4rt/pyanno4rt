"""AUC loss."""

# Author: Tim Ortkamp

# %% External package import

from sklearn.metrics import roc_auc_score

# %% Function definition


def auc_loss(true_labels, predicted_labels):
    """
    Compute the AUC loss.

    Parameters
    ----------
    true_labels : ndarray
        Ground truth label values.

    predicted_labels : ndarray
        Predicted label values.

    Returns
    -------
    float
        AUC loss value.
    """

    return -roc_auc_score(true_labels, predicted_labels)
