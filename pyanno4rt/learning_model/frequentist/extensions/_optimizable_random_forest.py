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
    random forest classifier. It exploits the pre-fitted structure of a \
    random forest to express the probability prediction function as the mean \
    function of all \
        :class:`~pyanno4rt.learning_model.frequentist._optimizable_decision_tree.OptimizableDecisionTree`
    objects from the member trees, and approximates a forest gradient as the \
    mean minimum input feature shift required to improve the prediction value.

    Attributes
    ----------
    members : list
        List with the optimizable decision tree members of the random forest.
    """

    def __init__(self):

        # Initialize the random forest members list
        self.members = []

    def initialize_members(
            self,
            forest):
        """
        Initialize the optimizable decision tree members of the random forest.

        Parameters
        ----------
        forest : object of class \
            :class:`sklearn.ensemble.RandomForestClassifier`
            The object used to represent the pre-fitted random forest.
        """

        # Loop over the members of the random forest
        for member in forest.estimators_:

            # Initialize the optimizable decision tree
            optimizable_tree = OptimizableDecisionTree()

            # Read the path information from the pre-fitted decision tree
            optimizable_tree.traverse(member)

            # Append the tree to the members list
            self.members.append(optimizable_tree)

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

            # Calculate the mean prediction value of all members
            prediction = mean([
                sub.predict_proba(features) for sub in self.members], axis=0)

            # Append the prediction value to the list
            predictions.append(prediction[0])

        else:

            # Loop over the samples in the feature array
            for sample in features:

                # Calculate the mean prediction value of all members
                prediction = mean([
                    sub.predict_proba(sample) for sub in self.members], axis=0)

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
        shifts = [sub.gradientize(features) for sub in self.members]

        return mean(shifts, axis=0)
