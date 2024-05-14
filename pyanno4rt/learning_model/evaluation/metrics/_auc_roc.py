"""AUC-ROC computation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pandas import DataFrame
from sklearn.metrics import roc_auc_score, roc_curve

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Function definition


def auc_roc(true_labels, predicted_labels):
    """
    Compute the AUC-ROC scores.

    Parameters
    ----------
    true_labels : ndarray
        Ground truth label values.

    predicted_labels : tuple
        Tuple of arrays with the training and out-of-folds labels predicted \
        by the machine learning outcome model.

    Returns
    -------
    scores : dict
        Dictionary with the training and out-of-folds AUC-ROC scores.
    """

    # Log a message about the AUC-ROC computation
    Datahub().logger.display_info("Computing AUC-ROC scores ...")

    # Initialize the AUC-ROC scores dictionary
    scores = {'Training': {'curve': None, 'value': None},
              'Out-of-folds': {'curve': None, 'value': None}}

    # Loop over the enumerated dictionary keys
    for i, source in enumerate(scores):

        # Enter the AUC-ROC curve into the dictionary
        scores[source]['curve'] = DataFrame(dict(zip(
            ('False Positive Rate', 'True Positive Rate', 'Threshold'),
            roc_curve(true_labels, predicted_labels[i]))))

        # Enter the AUC-ROC value into the dictionary
        scores[source]['value'] = roc_auc_score(
            true_labels, predicted_labels[i])

    return scores
