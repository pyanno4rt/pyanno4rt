"""ROC-AUC scores computation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pandas import DataFrame
from sklearn.metrics import roc_auc_score, roc_curve

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Class definition


class ROCScore():
    """
    ROC-AUC scores computation class.

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
            predicted_labels):
        """
        Compute the ROC-AUC (curve).

        Parameters
        ----------
        predicted_labels : tuple
            Tuple of arrays with the labels predicted by the learning model. \
            The first array holds the training prediction labels, the second \
            holds the out-of-folds prediction labels.

        Returns
        -------
        scores : dict
            Dictionary with the ROC-AUC scores. The keys are 'Training' and \
            'Out-of-folds', and for each a dataframe with false positive \
            rates, true positive rates, and thresholds is stored.

        auc_value : dict
            Dictionary with the AUC values. The keys are 'Training' and \
            'Out-of-folds', and for each a single AUC value is stored.
        """
        # Log a message about the ROC-AUC computation
        Datahub().logger.display_info("Computing model ROC-AUC scores ...")

        # Specify the evaluation sources
        sources = ('Training', 'Out-of-folds')

        # Specify the dataframe column names
        columns = ('False Positive Rate', 'True Positive Rate', 'Threshold')

        # Initialize the dictionaries for the ROC-AUC scores and the AUC value
        scores = {source: None for source in sources}
        auc_value = {source: None for source in sources}

        # Iterate over the number of prediction vectors
        for i, _ in enumerate(predicted_labels):

            # Create a dictionary with the true and predicted labels
            labels = dict(y_true=self.true_labels,
                          y_score=predicted_labels[i])

            # Compute the ROC-AUC scores
            scores[sources[i]] = DataFrame(
                dict(zip(columns, roc_curve(**labels))))

            # Compute the AUC value
            auc_value[sources[i]] = roc_auc_score(self.true_labels,
                                                  predicted_labels[i])

        return scores, auc_value
