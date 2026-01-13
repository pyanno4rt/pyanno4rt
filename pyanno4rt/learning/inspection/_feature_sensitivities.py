"""Feature sensitivities."""

# Author: Tim Ortkamp

# %% External package import

from numpy import atleast_2d, mean, std, vstack

# %% Internal package import

from pyanno4rt.logging import get_logger

# %% Function definition


def feature_sensitivities(model):
    """
    Compute the feature sensitivities.

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

    Returns
    -------
    dict
        Dictionary with the feature sensitivities and their uncertainties.
    """

    def calculate(features):
        """Calculate the sensitivities for a feature vector."""

        # Preprocess the feature vector
        preprocessed_features, _ = model.preprocess(features, None)

        # Get the preprocessing gradient
        preprocessing_gradient = model._preprocessing_gradient(features)

        # Get the predictor gradient
        predictor_gradient = model._predictor_gradient(preprocessed_features)

        return preprocessing_gradient @ predictor_gradient

    # Check if a holdout dataset is available
    if model.dataset.holdout_set is not None:

        # Get the holdout feature values
        features = model.dataset.holdout_set['feature_values']

    else:

        # Get the training feature values
        features = model.dataset.feature_values

    # Log a message about computing the feature sensitivities
    get_logger().info(
        "Computing feature sensitivities for '%s' ...", model.label)

    # Calculate the feature sensitivities
    sensitivities = vstack([calculate(atleast_2d(row)) for row in features])

    return {
        'expectations': mean(sensitivities, axis=0),
        'uncertainties': std(sensitivities, axis=0),
        'feature_names': model.dataset.feature_names}
