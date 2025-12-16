"""Model evaluation."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from numpy import where, zeros
from sklearn.model_selection import RepeatedStratifiedKFold

# %% Internal package import

from pyanno4rt.learning.evaluation import auc_pr, auc_roc, f1, kpi
from pyanno4rt.logging import get_logger
from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import validate_item, validate_type

# %% Class definition


class ModelEvaluator():
    """
    Model evaluation class.

    This class provides the computation method for a number of evaluation \
    metrics on a machine learning model.

    Parameters
    ----------
    splits : int, default=5
        Number of splits for the out-of-folds evaluation.

    repeats : int, default=1
        Number of repeats for the out-of-folds evaluation.

    Attributes
    ----------
    splits : int
        See 'Parameters'.

    repeats : int
        See 'Parameters'.

    results : dict
        Dictionary with the model evaluation results.
    """

    def __init__(
            self,
            splits,
            repeats):

        # Get the input arguments
        self.inputs = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.validate(self.inputs)

        # Log a message about the initialization of the model evaluator
        get_logger().info("Initializing model evaluator ...")

        # Get the input attributes
        self.splits = splits
        self.repeats = repeats

        # Initialize the results dictionary
        self.results = {}

    def to_dict(self):
        """Serialize the model evaluator into a dictionary."""

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """Deserialize the model evaluator from a dictionary."""

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

        else:

            # Get the training data
            features = model.dataset.feature_values
            labels = model.dataset.label_values

        # Log a message about the training prediction
        get_logger().info(
            "Evaluating training predictions for %s ...", model.label)

        # Get the training prediction
        training_prediction = model.predict(
            model.preprocess(features, labels)[0])

        # Log a message about the out-of-folds prediction
        get_logger().info(
            "Performing %s-fold cross-validation with %s repeat(s) to yield "
            "out-of-folds predictions for %s",
            self.splits, self.repeats, model.label)

        # Initialize the out-of-folds label prediction array
        oof_prediction = zeros((len(labels),))

        # Get the out-of-folds numbers
        folds = self.get_folds(features, labels)

        # Compute the returns across all repeats
        rep_returns = (map(compute_fold_labels, (
            (training_indices, validation_indices)
            for training_indices, validation_indices in (
                (where(folds[:, index] != number),
                 where(folds[:, index] == number))
                for index in range(folds.shape[1])
                for number in set(folds[:, index])))))

        # Loop over the returns
        for fold_indices, fold_labels in rep_returns:

            # Insert the fold labels at the fold indices
            oof_prediction[fold_indices] += fold_labels/folds.shape[1]

        #
        self.results['auc_pr'] = auc_pr(
            labels, (training_prediction, oof_prediction))

        #
        self.results['auc_roc'] = auc_roc(
            labels, (training_prediction, oof_prediction))

        #
        self.results['f1'] = f1(labels, (training_prediction, oof_prediction))

        #
        self.results['kpi'] = kpi(
            labels, (training_prediction, oof_prediction),
            tuple(
                self.results['f1'][source]['best']
                for source in ('Training', 'Out-of-folds')))

    def get_folds(
            self,
            features,
            labels):
        """
        Get the fold numbers for cross-validation.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : ndarray
            Values of the input labels.

        Returns
        -------
        ndarray
            Fold numbers.
        """

        # Clamp the number of splits
        clamped_n_splits = min(self.splits, sum(labels))

        # Initialize the stratified k-fold cross-validator
        cross_validator = RepeatedStratifiedKFold(
            n_splits=5 if clamped_n_splits == 1 else clamped_n_splits,
            n_repeats=self.repeats, random_state=3)

        # Get the stratification splits
        splits = tuple(cross_validator.split(features, labels))

        # Divide the splits into chunks (for each repeat)
        chunks = [
            splits[index:index+self.splits] for index in range(
                0, clamped_n_splits*self.repeats, clamped_n_splits)]

        # Initialize the fold numbers
        folds = zeros((len(labels), self.repeats))

        # Loop over the chunks
        for column, chunk in enumerate(chunks):

            # Loop over the chunk splits
            for number, (_, validation_index) in enumerate(chunk):

                # Enter the fold number for the validation set repetition
                folds[validation_index, column] = (
                    int(number) if self.splits != 1 else 1)

        return folds

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

        validation_map = {
            'splits': (
                partial(validate_type, options=int),
                partial(validate_item, reference=1, sign='>=')
                ),
            'repeats': (
                partial(validate_type, options=int),
                partial(validate_item, reference=1, sign='>=')
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
