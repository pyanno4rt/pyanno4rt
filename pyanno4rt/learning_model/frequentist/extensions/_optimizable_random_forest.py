"""Optimizable random forest."""

# %% External package import

from numpy import array, mean

# %% Internal package import

from pyanno4rt.learning_model.frequentist.extensions import (
    OptimizableDecisionTree)

# %% Class definition


class OptimizableRandomForest():
    """
    Optimizable random forest class.

    This class implements an optimizable surrogate model for scikit-learn's \
    random forest classifier. It exploits the pre-fitted structure of the \
    classifier to express the probability prediction function as the mean \
    prediction function value of all \
        :class:`~pyanno4rt.learning_model.frequentist._optimizable_decision_tree.OptimizableDecisionTree`
    objects from the subtrees, and approximates an input "gradient" as the \
    mean minimum input feature shift required to improve the prediction value.

    Attributes
    ----------
    subtrees : list
        List with the optimizable decision trees in the random forest.
    """

    def __init__(self):

        # Initialize the random forest subtree list
        self.subtrees = []

    def initialize_subtrees(
            self,
            forest):
        """
        Initialize the optimizable decision trees in the random forest.

        Parameters
        ----------
        forest : object of class \
            :class:`sklearn.ensemble.RandomForestClassifier`
            The object used to represent the pre-fitted random forest.
        """

        # Loop over the subtrees in the random forest
        for subtree in forest.estimators_:

            # Initialize the optimizable decision tree
            optimizable_tree = OptimizableDecisionTree()

            # Read the path information from the pre-fitted subtree
            optimizable_tree.traverse(subtree)

            # Append the optimizable tree to the subtree list
            self.subtrees.append(optimizable_tree)

    def predict_proba(
            self,
            features):
        """
        Predict the label values.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        Returns
        -------
        float or ndarray
            Value(s) of the predicted label(s).
        """

        # Initialize the list of predictions
        predictions = []

        # Check if the feature array has only a single row
        if features.shape[0] == 1:

            # Calculate the mean prediction value of all subtrees
            prediction = mean(
                [tree.predict_proba(features) for tree in self.subtrees],
                axis=0)

            # Append the prediction value to the list
            predictions.append(prediction[0])

        else:

            # Loop over the samples in the feature array
            for sample in features:

                # Calculate the mean prediction value of all subtrees
                prediction = mean(
                    [tree.predict_proba(sample) for tree in self.subtrees],
                    axis=0)

                # Append the prediction value to the list
                predictions.append(prediction[0])

        # Return the label predictions
        return array(predictions)

    def gradientize(
            self,
            features):
        """
        Gradientize the features with the minimum improvement shift.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        Returns
        -------
        ndarray
            Mean value of the minimum improvement shifts.
        """

        # Get the minimum shifts of all members
        shifts = [tree.gradientize(features) for tree in self.subtrees]

        return mean(shifts, axis=0)
