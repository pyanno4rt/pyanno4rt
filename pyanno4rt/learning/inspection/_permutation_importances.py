"""Permutation importance."""

# Author: Tim Ortkamp

# %% External package import

from numpy import atleast_1d, nan_to_num, unique, vstack, where
from sklearn.inspection import permutation_importance
from tensorflow.keras.utils import disable_interactive_logging

# %% Internal package import

import pyanno4rt.learning._maps as maps
from pyanno4rt.logging import get_logger

# %% Set package options

disable_interactive_logging()

# %% Function definition


def permutation_importances(model, score='AUC', permutations=20):
    """
    Compute the permutation importances.

    Parameters
    ----------
    model : object of class \
        :class:`~pyanno4rt.learning._models.forest._random_forest.RandomForest`\
        :class:`~pyanno4rt.learning._models.logistic._logistic_regression.LogisticRegression`\
        :class:`~pyanno4rt.learning._models.naive_bayes._naive_bayes.NaiveBayes`\
        :class:`~pyanno4rt.learning._models.neighbors._k_nearest_neighbors.KNearestNeighbors`\
        :class:`~pyanno4rt.learning._models.neural_network._feed_forward_net.FeedForwardNet`\
        :class:`~pyanno4rt.learning._models.svm._support_vector_machine.SupportVectorMachine`\
        :class:`~pyanno4rt.learning._models.tree._decision_tree.DecisionTree`
        The object used to represent the outcome model.

    score : {'AUC', 'BCE', 'Brier', 'Dice', 'Focal BCE', 'KLD', 'Hinge'}, \
        default='AUC'
        Permutation importance score.

    permutations : int, default=20
        Number of permutations.

    Returns
    -------
    dict
        Dictionary with the full and cross-validated permutation importances.
    """

    def compute_fold_importances(indices):
        """Compute the permutation importances for a single fold."""

        # Get the training and validation split
        split = [
            features[indices[0]], labels[indices[0]],
            features[indices[1]], labels[indices[1]]]

        # Fit the preprocessor on the training split
        model.fit_preprocessor(split[0], split[1])

        # Fit the predictor on the training split
        model.fit_predictor(*model.preprocess(split[0], split[1]))

        # Wrap the model
        wrapper_model = ModelWrapper(model)

        # Compute the permutation importance for the validation split
        importance = permutation_importance(
            wrapper_model, split[2], split[3], scoring=score_model,
            n_repeats=permutations, random_state=43)

        return nan_to_num(importance['importances'].T)

    def score_model(wrapper, features, true_labels):
        """Score a model's predictions."""

        # Transform the labels
        _, true_labels = wrapper.model.preprocess(features, true_labels)

        return scorer(true_labels, atleast_1d(wrapper.predict(features)))

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

    # Map the score labels to the score functions
    scorer = maps.LOSSES[score]

    # Wrap the model
    wrapped_model = ModelWrapper(model)

    # Log a message about computing the full data permutation importance
    get_logger().info(
        "Computing full data permutation importances for '%s' with %s "
        "permutations ...", model.label, permutations)

    # Compute the full data permutation importances
    full_importances = permutation_importance(
        wrapped_model, features, labels, scoring=score_model,
        n_repeats=permutations, random_state=42)['importances'].T

    # Log a message about computing the cross-validated permutation importances
    get_logger().info(
        "Computing cross-validated permutation importances for '%s' with %s "
        "permutations for %s splits and %s repeat(s) ...",
        model.label, permutations, len(unique(folds)), folds.shape[1])

    # Compute the cross-validated permutation importances
    cv_importances = map(compute_fold_importances, (
        (training_indices, validation_indices)
        for training_indices, validation_indices in (
                (where(folds[:, index] != number),
                 where(folds[:, index] == number))
                for index in range(folds.shape[1])
                for number in set(folds[:, index]))))

    return {
        'Full': full_importances,
        'Cross-validated': vstack(list(cv_importances)),
        'feature_names': model.dataset.feature_names,
        'score': score}


class ModelWrapper():
    """
    A wrapper class to run permutation importance with preprocessing.

    Parameters
    ----------
    model : object of class \
        :class:`~pyanno4rt.learning._models.forest._random_forest.RandomForest`\
        :class:`~pyanno4rt.learning._models.logistic._logistic_regression.LogisticRegression`\
        :class:`~pyanno4rt.learning._models.naive_bayes._naive_bayes.NaiveBayes`\
        :class:`~pyanno4rt.learning._models.neighbors._k_nearest_neighbors.KNearestNeighbors`\
        :class:`~pyanno4rt.learning._models.neural_network._feed_forward_net.FeedForwardNet`\
        :class:`~pyanno4rt.learning._models.svm._support_vector_machine.SupportVectorMachine`\
        :class:`~pyanno4rt.learning._models.tree._decision_tree.DecisionTree`
        The object used to represent the outcome model.
    """

    def __init__(self, model):

        # Get the model
        self.model = model

        # Set the fitting indicator
        self._is_fitted = True

    def fit(
            self,
            _,
            __):
        """
        Dummy fit required by permutation importance."""

        return self

    def predict(
            self,
            features):
        """
        Predict the label values.

        Parameters
        ----------
        features : ndarray
            Feature values.
        """

        # Transform the features
        preprocessed_features, _ = self.model.preprocess(features, None)

        return self.model.predict(preprocessed_features)
