"""Model evaluation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pyanno4rt.datahub import Datahub

# %% Internal package import

from pyanno4rt.learning_model.evaluation.metrics import (
    auc_pr, auc_roc, f1, kpi)

# %% Class definition


class ModelEvaluator():
    """
    Model evaluation class.

    This class provides the computation method for a number of evaluation \
    metrics on a machine learning outcome model.

    Parameters
    ----------
    model_label : str
        Label for the machine learning outcome model.

    Attributes
    ----------
    model_label: str
        See 'Parameters'.
    """

    def __init__(
            self,
            model_label):

        # Log a message about the initialization of the model evaluator
        Datahub().logger.display_info("Initializing model evaluator ...")

        # Get the model label from the arguments
        self.model_label = model_label

    def compute(
            self,
            true_labels,
            predicted_labels):
        """
        Compute the evaluation metrics.

        Parameters
        ----------
        true_labels : ndarray
            Ground truth label values.

        predicted_labels : tuple
            Tuple of arrays with the training and out-of-folds labels \
            predicted by the machine learning outcome model.
        """

        # Precompute the F1 scores
        f1_scores = f1(true_labels, predicted_labels)

        # Enter the model evaluation metrics into the datahub
        Datahub().model_evaluations[self.model_label] = {
            'auc_pr': auc_pr(true_labels, predicted_labels),
            'auc_roc': auc_roc(true_labels, predicted_labels),
            'f1': f1_scores,
            'kpi': kpi(true_labels, predicted_labels,
                       tuple(f1_scores[source]['best']
                             for source in ('Training', 'Out-of-folds')))}
