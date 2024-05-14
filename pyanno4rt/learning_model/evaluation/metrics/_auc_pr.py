"""AUC-PR computation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pandas import DataFrame
from sklearn.metrics import precision_recall_curve

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Function definition


def auc_pr(true_labels, predicted_labels):
    """
    Compute the AUC-PR scores.

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
        Dictionary with the training and out-of-folds AUC-PR scores.
    """

    # Log a message about the AUC-PR computation
    Datahub().logger.display_info("Computing AUC-PR scores ...")

    # Initialize the AUC-PR scores dictionary
    scores = {'Training': None, 'Out-of-folds': None}

    # Loop over the enumerated dictionary keys
    for i, source in enumerate(scores):

        # Compute the AUC-PR scores
        precision, recall, _ = precision_recall_curve(
            true_labels, predicted_labels[i])

        # Enter the AUC-PR scores into the dictionary
        scores[source] = DataFrame({'Precision': precision, 'Recall': recall})

    return scores
