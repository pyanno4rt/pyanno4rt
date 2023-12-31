"""Model evaluation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pyanno4rt.datahub import Datahub

# %% Internal package import

from pyanno4rt.learning_model.evaluation.metrics import (
    F1Score, ModelKPI, PRScore, ROCScore)

# %% Class definition


class ModelEvaluator():
    """
    Model evaluation class.

    This class provides a collection of evaluation metrics to be computed in \
    a single method call.

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

    evaluations : dict
        Dictionary with the evaluation metrics.
    """

    def __init__(
            self,
            model_name,
            true_labels):

        # Log a message about the initialization of the model evaluator
        Datahub().logger.display_info("Initializing model evaluator ...")

        # Get the instance attributes from the arguments
        self.model_name = model_name
        self.true_labels = true_labels

        # Initialize the dictionary with the evaluations
        self.evaluations = {}

    def compute(
            self,
            predicted_labels):
        """
        Compute the evaluation metrics.

        Parameters
        ----------
        predicted_labels : tuple
            Tuple of arrays with the labels predicted by the learning model. \
            The first array holds the training prediction labels, the second \
            holds the out-of-folds prediction labels.
        """
        # Compute the AUC-ROC scores
        auc_roc = ROCScore(self.model_name, self.true_labels).compute(
            predicted_labels)

        # Compute the AUC-PR scores
        auc_pr = PRScore(self.model_name, self.true_labels).compute(
            predicted_labels)

        # Compute the F1 scores
        f1 = F1Score(self.model_name, self.true_labels).compute(
            predicted_labels)

        # Compute the Model KPIs
        kpi = ModelKPI(self.model_name, self.true_labels).compute(
            predicted_labels, tuple(f1[1].values()))

        # Update the evaluations dictionary with the metrics results
        self.evaluations.update({'indicators': kpi,
                                 'auc_roc': auc_roc,
                                 'auc_pr': auc_pr,
                                 'f1': f1})
