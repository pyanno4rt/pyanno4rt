"""Model KPI computation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.metrics import (accuracy_score, brier_score_loss,
                             cohen_kappa_score, f1_score, hamming_loss,
                             jaccard_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from tensorflow.compat.v2.keras.losses import binary_crossentropy

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Class definition


class ModelKPI():
    """
    Model KPI computation class.

    Parameters
    ----------
    model_name : string
        Name of the learning model.

    true_labels : ndarray
        Ground truth values for the labels to predict.

    Attributes
    ----------
    model_name : string
        See 'Parameters'.

    true_labels : ndarray
        See 'Parameters'.
    """

    def __init__(
            self,
            model_name,
            true_labels):

        # Get the instance attributes from the arguments
        self.model_name = model_name
        self.true_labels = true_labels

    def compute(
            self,
            predicted_labels,
            thresholds=(0.5, 0.5)):
        """
        Compute the KPIs.

        Parameters
        ----------
        predicted_labels : tuple
            Tuple of arrays with the labels predicted by the learning model. \
            The first array holds the training prediction labels, the second \
            holds the out-of-folds prediction labels.

        thresholds : tuple, default=(0.5, 0.5)
            Probability thresholds for the binarization of the probability \
            predictions.

        Returns
        -------
        indicators : dict
            Dictionary with the key performance indicators. The keys are \
            'Training' and 'Out-of-folds', and for each a dictionary with \
            indicator/value pairs is stored.
        """
        # Log a message about the KPI computation
        Datahub().logger.display_info("Computing model KPIs ...")

        # Binarize and make the predictions iterable
        binary_prediction = tuple(
            predicted_labels[i] >= thresholds[i] if thresholds[i] == 1
            else predicted_labels[i] > thresholds[i]
            for i, _ in enumerate(predicted_labels))

        # Specify the evaluation sources
        sources = ('Training', 'Out-of-folds')

        # Specify the indicator names
        indicator_names = ('Logloss', 'Brier score', 'Subset accuracy',
                           'Cohen Kappa', 'Hamming loss', 'Jaccard score',
                           'Precision', 'Recall', 'F1 score', 'MCC', 'AUC')

        # Initialize the indicator dictionary
        indicators = {source: {
            name: None for name in indicator_names} for source in sources}

        # Iterate over the number of prediction vectors
        for i, _ in enumerate(predicted_labels):

            # Compute the log loss
            indicators[sources[i]]['Logloss'] = binary_crossentropy(
                self.true_labels, predicted_labels[i]).numpy().mean()

            # Compute the Brier score
            indicators[sources[i]]['Brier score'] = brier_score_loss(
                self.true_labels, predicted_labels[i])

            # Compute the (subset) accuracy
            indicators[sources[i]]['Subset accuracy'] = accuracy_score(
                self.true_labels, binary_prediction[i])

            # Compute Cohen's Kappa
            indicators[sources[i]]['Cohen Kappa'] = cohen_kappa_score(
                self.true_labels, binary_prediction[i])

            # Compute the Hamming loss
            indicators[sources[i]]['Hamming loss'] = hamming_loss(
                self.true_labels, binary_prediction[i])

            # Compute the Jaccard score
            indicators[sources[i]]['Jaccard score'] = jaccard_score(
                self.true_labels, binary_prediction[i])

            # Compute the precision score
            indicators[sources[i]]['Precision'] = precision_score(
                self.true_labels, binary_prediction[i])

            # Compute the recall score
            indicators[sources[i]]['Recall'] = recall_score(
                self.true_labels, binary_prediction[i])

            # Compute the F1 score
            indicators[sources[i]]['F1 score'] = f1_score(
                self.true_labels, binary_prediction[i])

            # Compute the F1 score
            indicators[sources[i]]['MCC'] = matthews_corrcoef(
                self.true_labels, binary_prediction[i])

            # Compute the AUC score
            indicators[sources[i]]['AUC'] = roc_auc_score(
                self.true_labels, predicted_labels[i])

        return indicators
