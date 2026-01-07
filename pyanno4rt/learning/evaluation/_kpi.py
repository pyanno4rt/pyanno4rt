"""KPI computation."""

# Author: Tim Ortkamp

# %% External package import

from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    cohen_kappa_score, f1_score, hamming_loss, jaccard_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from tensorflow.keras.losses import binary_crossentropy

# %% Internal package import

from pyanno4rt.logging import get_logger

# %% Function definition


def kpi(true_labels, predicted_labels, thresholds=(0.5, 0.5)):
    """
    Compute the model KPIs.

    Parameters
    ----------
    true_labels : ndarray
        Ground truth label values.

    predicted_labels : tuple
        Arrays with the predicted full data and out-of-folds labels.

    thresholds : tuple, default=(0.5, 0.5)
        Probability thresholds for binarization.

    Returns
    -------
    scores : dict
        Dictionary with the full data and out-of-folds KPIs.
    """

    # Log a message about the KPI computation
    get_logger().info("Computing KPIs ...")

    # Initialize the KPI dictionary
    scores = {
        source: {
            kpi: None for kpi in (
                'Log loss', 'Brier score', 'Balanced accuracy', 'Cohen Kappa',
                'Hamming loss', 'Jaccard score', 'Precision', 'Recall',
                'F1', 'MCC', 'AUC-PR', 'AUC-ROC')
            }
        for source in ('Full', 'Cross-validated')}

    # Binarize the predicted labels
    binarized = tuple(
        labels >= thresholds[index] if thresholds[index] == 1
        else labels > thresholds[index]
        for index, labels in enumerate(predicted_labels))

    # Loop over the dictionary elements
    for index, source in enumerate(scores):

        # Compute the log loss
        scores[source]['Log loss'] = binary_crossentropy(
            true_labels, predicted_labels[index]).numpy().mean()

        # Compute the Brier score
        scores[source]['Brier score'] = brier_score_loss(
            true_labels, predicted_labels[index])

        # Compute the (subset) accuracy
        scores[source]['Balanced accuracy'] = balanced_accuracy_score(
            true_labels, binarized[index])

        # Compute Cohen's Kappa
        scores[source]['Cohen Kappa'] = cohen_kappa_score(
            true_labels, binarized[index])

        # Compute the Hamming loss
        scores[source]['Hamming loss'] = hamming_loss(
            true_labels, binarized[index])

        # Compute the Jaccard score
        scores[source]['Jaccard score'] = jaccard_score(
            true_labels, binarized[index])

        # Compute the precision score
        scores[source]['Precision'] = precision_score(
            true_labels, binarized[index])

        # Compute the recall score
        scores[source]['Recall'] = recall_score(true_labels, binarized[index])

        # Compute the F1 score
        scores[source]['F1'] = f1_score(true_labels, binarized[index])

        # Compute the Matthews correlation
        scores[source]['MCC'] = matthews_corrcoef(
            true_labels, binarized[index])

        # Compute the AUC-PR score
        scores[source]['AUC-PR'] = average_precision_score(
            true_labels, predicted_labels[index])

        # Compute the AUC-ROC score
        scores[source]['AUC-ROC'] = roc_auc_score(
            true_labels, predicted_labels[index])

    return scores
