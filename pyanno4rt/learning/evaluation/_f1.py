"""F1 computation."""

# Author: Tim Ortkamp

# %% External package import

from pandas import Series
from sklearn.metrics import f1_score, precision_recall_curve

# %% Internal package import

from pyanno4rt.logging import get_logger

# %% Function definition


def f1(true_labels, predicted_labels):
    """
    Compute the F1 scores.

    Parameters
    ----------
    true_labels : ndarray
        Ground truth label values.

    predicted_labels : tuple
        Arrays with the predicted full data and out-of-folds labels.

    Returns
    -------
    scores : dict
        Dictionary with the full data and out-of-folds F1 scores and the \
        location of the best score.
    """

    # Log a message about the F1 computation
    get_logger().info("Computing F1 scores ...")

    # Initialize the F1 scores dictionary
    scores = {
        'Full': {'values': None, 'best': None},
        'Cross-validated': {'values': None, 'best': None}}

    # Loop over the dictionary elements
    for index, source in enumerate(scores):

        # Get the thresholds from the PR curve
        _, _, thresholds = precision_recall_curve(
            true_labels, predicted_labels[index])

        # Store the F1 values
        scores[source]['values'] = Series(
            {threshold: f1_score(
                true_labels, predicted_labels[index] > threshold)
             for threshold in thresholds})

        # Store the location of the maximum F1 value
        scores[source]['best'] = scores[source]['values'].idxmax()

    return scores
