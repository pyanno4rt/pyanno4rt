"""Projection forest."""

# Author: Tim Ortkamp

# %% External package import

from numpy import mean

# %% Internal package import

from pyanno4rt.learning.models.tree import ProjectionTree

# %% Class definition


class ProjectionForest():
    """
    Projection forest class.

    This class implements a differentiable random forest using a \
    projection-based gradient approximation.

    Attributes
    ----------
    forest : None or object of class \
        :class:`sklearn.ensemble.RandomForestClassifier`
        The object used to represent the "hard", non-differentiable random \
        forest.

    projection_trees : None or object of class \
        :class:`pyanno4rt.learning.models.tree._projection_tree.ProjectionTree`
        The object used to represent the projection-based tree approximations.
    """

    def __init__(self):

        # Initialize the random forest
        self.forest = None

        # Initialize the projection trees
        self.projection_trees = None

    def parse(
            self,
            forest):
        """
        Parse a random forest.

        Parameters
        ----------
        forest : object of class \
            :class:`sklearn.ensemble.RandomForestClassifier`
            The object used to represent the "hard", non-differentiable \
            random forest.
        """

        # Get the random forest
        self.forest = forest

        # Get the projection trees
        self.projection_trees = [ProjectionTree() for _ in forest.estimators_]

        # Loop over the estimators
        for projector, estimator in zip(
                self.projection_trees, forest.estimators_):

            # Parse the estimators
            projector.parse(estimator)

    def predict_proba(
            self,
            features):
        """
        Predict the label value(s).

        Parameters
        ----------
        features : ndarray
            Feature values.

        Returns
        -------
        float or ndarray
            Predicted label value(s).
        """

        # Check if the feature array has only a single row
        if features.shape[0] == 1:

            # Return a single label prediction value
            return self.forest.predict_proba(features)[0][1]

        # Else, return an array with label predictions
        return self.forest.predict_proba(features)[:, 1]

    def gradientize(
            self,
            features,
            epsilon=1e-12):
        """
        Calculate the random forest gradient.

        Parameters
        ----------
        features : ndarray
            Feature values.

        epsilon : float, default=1e-12
            Perturbation value to ensure interior-point shifts.

        Returns
        -------
        ndarray
            Gradient w.r.t the features.
        """

        # Get the estimator gradients
        gradients = [
            tree.gradientize(features, epsilon)
            for tree in self.projection_trees]

        return mean(gradients, axis=0)
