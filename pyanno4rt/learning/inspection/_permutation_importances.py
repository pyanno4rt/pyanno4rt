"""Permutation importance computation."""

# Author: Tim Ortkamp

# %% External package import

from numpy import vstack, where, zeros
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold

# %% Internal package import

import pyanno4rt.learning._maps as maps
from pyanno4rt.logging import get_logger

# %% Function definition


def permutation_importances(
        model, score='AUC', permutations=20, splits=5, repeats=1):
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

    splits : int, default=5
        Number of splits for cross-validated evaluation.

    repeats : int, default=1
        Number of repeats for cross-validated evaluation.

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

    else:

        # Get the training data
        features = model.dataset.feature_values
        labels = model.dataset.label_values

    # Map the score labels to the score functions
    scorer = maps.LOSSES[score]

    # Compute the full data permutation importances
    full_importances = permutation_importance(
        model.predictor, *model.preprocessor.fit_transform(features, labels),
        scoring=score_model, n_repeats=permutations, random_state=42)[
            'importances'].T

    # Log a message about computing the cross-validated permutation importances
    get_logger().info(
        "Computing cross-validated permutation importances for '%s' with %s "
        "permutations for %s splits and %s repeats ...",
        model.label, permutations, splits, repeats)

    # Clamp the number of splits
    clamped_n_splits = min(splits, sum(labels))

    # Initialize the stratified k-fold cross-validator
    cross_validator = RepeatedStratifiedKFold(
        n_splits=5 if clamped_n_splits == 1 else clamped_n_splits,
        n_repeats=repeats, random_state=4)

    # Get the stratification splits
    stratifications = tuple(cross_validator.split(features, labels))

    # Divide the splits into chunks (for each repeat)
    chunks = [
        stratifications[index:index+splits] for index in range(
            0, clamped_n_splits*repeats, clamped_n_splits)]

    # Initialize the fold numbers
    folds = zeros((len(labels), repeats))

    # Loop over the chunks
    for column, chunk in enumerate(chunks):

        # Loop over the chunk splits
        for number, (_, validation_index) in enumerate(chunk):

            # Enter the fold number for the validation set repetition
            folds[validation_index, column] = (
                int(number) if splits != 1 else 1)

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
