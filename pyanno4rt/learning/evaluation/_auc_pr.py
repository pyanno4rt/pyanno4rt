"""AUC-PR computation."""

# Author: Tim Ortkamp

# %% External package import

from pandas import DataFrame
from sklearn.metrics import precision_recall_curve

# %% Internal package import

from pyanno4rt.logging import get_logger

# %% Function definition


def auc_pr(true_labels, predicted_labels):
    """
    Compute the AUC-PR scores.

    Parameters
    ----------
    true_labels : ndarray
        Ground truth label values.

    predicted_labels : tuple
        Arrays with the predicted full data and out-of-folds labels.

    Returns
    -------
    scores : dict
        Dictionary with the full data and out-of-folds AUC-PR scores.
    """

    # Log a message about the AUC-PR computation
    get_logger().info("Computing AUC-PR scores ...")

    # Initialize the AUC-PR scores dictionary
    scores = {'Full': None, 'Cross-validated': None}

    # Loop over the dictionary elements
    for index, source in enumerate(scores):

        # Compute the AUC-PR scores
        precision, recall, _ = precision_recall_curve(
            true_labels, predicted_labels[index])

        # Store the AUC-PR scores
        scores[source] = DataFrame({'Precision': precision, 'Recall': recall})

    return scores
