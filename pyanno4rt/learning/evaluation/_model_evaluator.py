"""Model evaluation."""

# Author: Tim Ortkamp

# %% External package import

from copy import deepcopy
from numpy import unique, where, zeros

# %% Internal package import

from pyanno4rt.learning.evaluation import auc_pr, auc_roc, f1, kpi
from pyanno4rt.logging import get_logger
from pyanno4rt.tools import filter_dict

# %% Class definition


class ModelEvaluator():
    """
    Model evaluation class.

    This class provides methods to evaluate an outcome model.

    Attributes
    ----------
    results : dict
        Dictionary with the model evaluation results.
    """

    def __init__(self):

        # Get the input arguments
        self.inputs = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.validate(self.inputs)

        # Log a message about the initialization of the model evaluator
        get_logger().info("Initializing model evaluator ...")

        # Initialize the results dictionary
        self.results = {}

    def to_dict(self):
        """Serialize the model evaluator into a dictionary."""

        # Get the parameter dictionary
        dictionary = deepcopy(self.inputs)

        return dictionary

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the model evaluator from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the model evaluator parameters.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.evaluation._model_evaluator.ModelEvaluator`
            The object used to represent the model evaluator.
        """

        return cls(**dictionary)

    def evaluate(
            self,
            true_labels,
            predicted_labels):
        """
        Evaluate the performance metrics.

        Parameters
        ----------
        true_labels : ndarray
            Ground truth labels.

        predicted_labels : tuple
            Arrays with the predicted full data and out-of-folds labels.
        """

        # Calculate the PR-AUC scores
        self.results['auc_pr'] = auc_pr(true_labels, predicted_labels)

        # Calculate the ROC-AUC scores
        self.results['auc_roc'] = auc_roc(true_labels, predicted_labels)

        # Calculate the F1 scores
        self.results['f1'] = f1(true_labels, predicted_labels)

        # Calculate the KPIs
        self.results['kpi'] = kpi(
            true_labels, predicted_labels,
            (self.results['f1']['Full']['best'],
             self.results['f1']['Cross-validated']['best']))

    def run(
            self,
            model):
        """
        Evaluate a model.

        Parameters
        ----------
        model : object of class \
            :class:`~pyanno4rt.learning._models.forest._random_forest.RandomForest`\
            :class:`~pyanno4rt.learning._models.logistic._logistic_regression.LogisticRegression`\
            :class:`~pyanno4rt.learning._models.naive_bayes._naive_bayes.NaiveBayes`\
            :class:`~pyanno4rt.learning._models.neighbors._k_nearest_neighbors.KNearestNeighbors`\
            :class:`~pyanno4rt.learning._models.network._neural_network.NeuralNetwork`\
            :class:`~pyanno4rt.learning._models.svm._support_vector_machine.SupportVectorMachine`\
            :class:`~pyanno4rt.learning._models.tree._decision_tree.DecisionTree`
            The object used to represent the outcome model.
        """

        def compute_fold_labels(indices):
            """Compute the holdout labels for a single fold."""

            # Get the training and validation split
            split = [
                features[indices[0]], labels[indices[0]],
                features[indices[1]]]

            # Fit the preprocessor on the training split
            model.fit_preprocessor(split[0], split[1])

            # Transform both splits
            split = [
                *model.preprocess(split[0], split[1]),
                *model.preprocess(split[2], None)]

            # Fit the predictor on the training split
            model.fit_predictor(split[0], split[1])

            return (indices[1], model.predict(split[2]))

        # Check if a holdout dataset is available
        if model.dataset.holdout_set is not None:

            # Get the holdout data
            features = model.dataset.holdout_set['feature_values']
            labels = model.dataset.holdout_set['label_values']
            folds = model.dataset.holdout_set['folds']

        else:

            # Get the training data
            features = model.dataset.feature_values
            labels = model.dataset.label_values
            folds = model.dataset.folds

        # Log a message about the full data prediction
        get_logger().info(
            "Yielding full data predictions for '%s' ...", model.label)

        # Get the full data prediction
        full_prediction = model.predict(model.preprocess(features, labels)[0])

        # Log a message about the out-of-folds prediction
        get_logger().info(
            "Performing %s-fold cross-validation with %s repeat(s) to yield "
            "out-of-folds predictions for '%s' ...",
            len(unique(folds)), folds.shape[1], model.label)

        # Initialize the array for the out-of-folds predictions
        cv_prediction = zeros((len(labels),))

        # Get the repeated cross-validation returns
        cv_returns = (map(compute_fold_labels, (
            (training_indices, validation_indices)
            for training_indices, validation_indices in (
                (where(folds[:, index] != number),
                 where(folds[:, index] == number))
                for index in range(folds.shape[1])
                for number in set(folds[:, index])))))

        # Loop over the returns
        for fold_indices, fold_labels in cv_returns:

            # Insert the fold labels
            cv_prediction[fold_indices] += fold_labels/folds.shape[1]

        # Evaluate the model predictions
        self.evaluate(labels, (full_prediction, cv_prediction))

    def validate(
            self,
            inputs):
        """
        Validate the input arguments.

        Parameters
        ----------
        inputs : dict
            Dictionary with the mappings between argument names and values.
        """

        validation_map = {}

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
