"""KPI computation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from sklearn.metrics import (
    accuracy_score, brier_score_loss, cohen_kappa_score, f1_score,
    hamming_loss, jaccard_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score)
from tensorflow.keras.losses import binary_crossentropy

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Function definition


def kpi(true_labels, predicted_labels, thresholds=(0.5, 0.5)):
    """
    Compute the model KPIs.

    Parameters
    ----------
    true_labels : ndarray
        Ground truth label values.

    predicted_labels : tuple
        Tuple of arrays with the training and out-of-folds labels predicted \
        by the machine learning outcome model.

    thresholds : tuple, default=(0.5, 0.5)
        Probability thresholds for the binarization of the probability \
        predictions.

    Returns
    -------
    scores : dict
        Dictionary with the training and out-of-folds KPIs.
    """

    # Log a message about the KPI computation
    Datahub().logger.display_info("Computing KPIs ...")

    # Binarize the predicted labels
    binaries = tuple(
        label >= thresholds[i] if thresholds[i] == 1 else label > thresholds[i]
        for i, label in enumerate(predicted_labels))

    # Initialize the KPI dictionary
    scores = {source: {kpi: None for kpi in (
        'Logloss', 'Brier score', 'Subset accuracy', 'Cohen Kappa',
        'Hamming loss', 'Jaccard score', 'Precision', 'Recall', 'F1 score',
        'MCC', 'AUC')} for source in ('Training', 'Out-of-folds')}

    # Loop over the enumerated dictionary keys
    for i, source in enumerate(scores):

        # Compute the log loss
        scores[source]['Logloss'] = binary_crossentropy(
            true_labels, predicted_labels[i]).numpy().mean()

        # Compute the Brier score
        scores[source]['Brier score'] = brier_score_loss(
            true_labels, predicted_labels[i])

        # Compute the (subset) accuracy
        scores[source]['Subset accuracy'] = accuracy_score(
            true_labels, binaries[i])

        # Compute Cohen's Kappa
        scores[source]['Cohen Kappa'] = cohen_kappa_score(
            true_labels, binaries[i])

        # Compute the Hamming loss
        scores[source]['Hamming loss'] = hamming_loss(true_labels, binaries[i])

        # Compute the Jaccard score
        scores[source]['Jaccard score'] = jaccard_score(
            true_labels, binaries[i])

        # Compute the precision score
        scores[source]['Precision'] = precision_score(true_labels, binaries[i])

        # Compute the recall score
        scores[source]['Recall'] = recall_score(true_labels, binaries[i])

        # Compute the F1 score
        scores[source]['F1 score'] = f1_score(true_labels, binaries[i])

        # Compute the Matthews correlation
        scores[source]['MCC'] = matthews_corrcoef(true_labels, binaries[i])

        # Compute the AUC score
        scores[source]['AUC'] = roc_auc_score(true_labels, predicted_labels[i])

    return scores
