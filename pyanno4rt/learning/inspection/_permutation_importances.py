"""Permutation importance computation."""

# Author: Tim Ortkamp

# %% External package import

from numpy import unique, vstack, where
from sklearn.inspection import permutation_importance

# %% Internal package import

import pyanno4rt.learning._maps as maps
from pyanno4rt.logging import get_logger

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
        :class:`~pyanno4rt.learning._models.network._neural_network.NeuralNetwork`\
        :class:`~pyanno4rt.learning._models.svm._support_vector_machine.SupportVectorMachine`\
        :class:`~pyanno4rt.learning._models.tree._decision_tree.DecisionTree`
        The object used to represent the outcome model.

    score : {'AUC', 'Brier score', 'Logloss'}, default='AUC'
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

        # Transform both splits
        split = [
            *model.preprocess(split[0], split[1]),
            *model.preprocess(split[2], split[3])]

        # Fit the predictor on the training split
        model.fit_predictor(split[0], split[1])

        # Compute the permutation importance for the validation split
        importance = permutation_importance(
            model.predictor, split[2], split[3], scoring=score_model,
            n_repeats=permutations, random_state=43)

        return importance['importances'].T

    def score_model(model, features, true_labels):
        """Score a model's prediction."""

        return scorer(true_labels, model.predict(features))

    # Log a message about computing the full data permutation importance
    get_logger().info(
        "Computing full data permutation importances for '%s' with %s "
        "permutations ...", model.label, permutations)

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

    # Check if a preprocessor has been provided
    if model.preprocessor is not None:

        # Fit the preprocessor
        model.fit_preprocessor(features, labels)

    # Compute the full data permutation importances
    full_importances = permutation_importance(
        model.predictor, *model.preprocess(features, labels),
        scoring=score_model, n_repeats=permutations, random_state=42)[
            'importances'].T

    # Log a message about computing the cross-validated permutation importances
    get_logger().info(
        "Computing cross-validated permutation importances for '%s' with %s "
        "permutations for %s splits and %s repeats ...",
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
