"""F1 scores computation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pandas import Series
from sklearn.metrics import f1_score, precision_recall_curve

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Class definition


class F1Score():
    """
    F1 metric computation class.

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
        Compute F1 scores.

        Parameters
        ----------
        predicted_labels : tuple
            Tuple of arrays with the labels predicted by the learning model. \
            The first array holds the training prediction labels, the second \
            holds the out-of-folds prediction labels.

        Returns
        -------
        f1_scores : dict
            Dictionary with the F1 scores for different thresholds. The keys \
            are 'Training' and 'Out-of-folds', and for each a series of \
            threshold/F1 value pairs is stored.

        best_f1 : dict
            Location of the best F1 score. The keys are 'Training' and \
            'Out-of-folds', and for each a single threshold value is stored
            which refers to the maximum F1 score.
        """
        # Log a message about the F1 computation
        Datahub().logger.display_info("Computing model F1 scores ...")

        # Specify the evaluation sources
        sources = ('Training', 'Out-of-folds')

        # Initialize the dictionaries for the F1 scores and the best F1 score
        f1_scores = {source: None for source in sources}
        best_f1 = {source: None for source in sources}

        # Iterate over the number of prediction vectors
        for i in range(len(predicted_labels)):

            # Get the threshold values from the precision-recall curve
            _, _, thresholds = precision_recall_curve(self.true_labels,
                                                      predicted_labels[i])

            # Compute the F1 scores
            f1_scores[sources[i]] = Series(
                {threshold: f1_score(self.true_labels,
                                     predicted_labels[i] > threshold)
                 for threshold in thresholds})

            # Compute the location of the maximum F1 score
            best_f1[sources[i]] = f1_scores[sources[i]].idxmax()

        return f1_scores, best_f1
