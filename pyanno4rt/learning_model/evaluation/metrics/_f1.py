"""F1 computation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pandas import Series
from sklearn.metrics import f1_score, precision_recall_curve

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Function definition


def f1(true_labels, predicted_labels):
    """
    Compute the F1 scores.

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
        Dictionary with the training and out-of-folds F1 scores and the \
        location of the best score.
    """

    # Log a message about the F1 computation
    Datahub().logger.display_info("Computing F1 scores ...")

    # Initialize the F1 scores dictionary
    scores = {'Training': {'values': None, 'best': None},
              'Out-of-folds': {'values': None, 'best': None}}

    # Loop over the enumerated dictionary keys
    for i, source in enumerate(scores):

        # Get the threshold values from the precision-recall curve
        _, _, thresholds = precision_recall_curve(
            true_labels, predicted_labels[i])

        # Enter the F1 values into the dictionary
        scores[source]['values'] = Series(
            {threshold: f1_score(true_labels, predicted_labels[i] > threshold)
             for threshold in thresholds})

        # Enter the maximum F1 value index into the dictionary
        scores[source]['best'] = scores[source]['values'].idxmax()

    return scores
