"""Precision-Recall scores computation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pandas import DataFrame
from sklearn.metrics import precision_recall_curve

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Class definition


class PRScore():
    """
    Precision-Recall scores computation class.

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
        Compute the precision and the recall (curves).

        Parameters
        ----------
        predicted_labels : tuple
            Tuple of arrays with the labels predicted by the learning model. \
            The first array holds the training prediction labels, the second \
            holds the out-of-folds prediction labels.

        Returns
        -------
        precision_recall : dict
            Dictionary with the precision-recall scores. The keys are \
            'Training' and 'Out-of-folds', and for each a dataframe with \
            precision-recall scores is stored.
        """
        # Log a message about the AUC-PR computation
        Datahub().logger.display_info("Computing model AUC-PR scores ...")

        # Specify the evaluation sources
        sources = ('Training', 'Out-of-folds')

        # Initialize the AUC-PR scores dictionary
        precision_recall = {source: None for source in sources}

        # Iterate over the number of prediction vectors
        for i, _ in enumerate(predicted_labels):

            # Compute the precision-recall scores
            precision, recall, _ = precision_recall_curve(self.true_labels,
                                                          predicted_labels[i])

            # Add a dataframe with the scores to the dictionary
            precision_recall[sources[i]] = DataFrame({'Precision': precision,
                                                      'Recall': recall})

        return precision_recall
