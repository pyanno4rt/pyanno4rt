"""AUC-ROC computation."""

# Author: Tim Ortkamp

# %% External package import

from pandas import DataFrame
from sklearn.metrics import roc_auc_score, roc_curve

# %% Internal package import

from pyanno4rt.logging import get_logger

# %% Function definition


def auc_roc(true_labels, predicted_labels):
    """
    Compute the AUC-ROC scores.

    Parameters
    ----------
    true_labels : ndarray
        Ground truth label values.

    predicted_labels : tuple
        Arrays with the predicted full data and out-of-folds labels.

    Returns
    -------
    scores : dict
        Dictionary with the full data and out-of-folds AUC-ROC scores.
    """

    # Log a message about the AUC-ROC computation
    get_logger().info("Computing AUC-ROC scores ...")

    # Initialize the AUC-ROC scores dictionary
    scores = {
        'Full': {'curve': None, 'value': None},
        'Cross-validated': {'curve': None, 'value': None}}

    # Loop over the dictionary elements
    for index, source in enumerate(scores):

        # Compute the AUC-ROC curve points
        scores[source]['curve'] = DataFrame(dict(zip(
            ('False Positive Rate', 'True Positive Rate', 'Threshold'),
            roc_curve(true_labels, predicted_labels[index]))))

        # Compute the AUC-ROC value
        scores[source]['value'] = roc_auc_score(
            true_labels, predicted_labels[index])

    return scores
