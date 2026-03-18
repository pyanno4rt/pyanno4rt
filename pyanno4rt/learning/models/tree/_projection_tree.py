"""Projection tree."""

# Author: Tim Ortkamp

# %% External package import

from math import inf

from numpy import clip, tile, zeros
from numpy.linalg import norm as npnorm

# %% Class definition


class ProjectionTree():
    """
    Projection tree class.

    This class implements a differentiable decision tree using a \
    projection-based gradient approximation.

    Attributes
    ----------
    tree : None or object of class \
        :class:`sklearn.tree.DecisionTreeClassifier`
        The object used to represent the "hard", non-differentiable tree.

    bounds : None or dict
        Feature bounds for each leaf.

    outcomes : None or dict
        Outcome value for each leaf.
    """

    def __init__(self):

        # Initialize the decision tree
        self.tree = None

        # Initialize the leaf bounds and outcomes
        self.bounds, self.outcomes = None, None

    def parse(
            self,
            tree):
        """
        Parse a decision tree.

        Parameters
        ----------
        tree : object of class \
            :class:`sklearn.tree.DecisionTreeClassifier`
            The object used to represent the "hard", non-differentiable tree.
        """

        def get_bounds(tree):
            """Get the leaf bounds."""

            # Initialize the leaf bound dictionary
            leaf_bounds = {}

            # Get the tree structure
            structure = tree.tree_

            # Initialize the pending nodes with the root
            nodes_to_visit = [(0, tile([-inf, inf], (tree.n_features_in_, 1)))]

            # Loop while nodes visits are pending
            while nodes_to_visit:

                # Get the most recent node
                node_id, bounds = nodes_to_visit.pop()

                # Check if the node is a leaf
                if structure.feature[node_id] == -2:

                    # Add the current bounds
                    leaf_bounds[node_id] = bounds

                    # Continue with the next node
                    continue

                # Get the split feature index
                index = structure.feature[node_id]

                # Get the split feature threshold
                threshold = structure.threshold[node_id]

                # Copy the current bounds
                right_child_bounds = bounds.copy()
                left_child_bounds = bounds.copy()

                # Update the bounds for the right child
                right_child_bounds[index, 0] = max(
                    right_child_bounds[index, 0], threshold)

                # Update the bounds for the left child
                left_child_bounds[index, 1] = min(
                    left_child_bounds[index, 1], threshold)

                # Extend the pending nodes
                nodes_to_visit.extend([
                    (structure.children_right[node_id], right_child_bounds),
                    (structure.children_left[node_id], left_child_bounds)])

            return leaf_bounds

        # Get the decision tree
        self.tree = tree

        # Get the leaf bounds
        self.bounds = get_bounds(tree)

        # Get the leaf outcomes
        self.outcomes = {
            index: tree.tree_.value[index, 0, 1]
            for index in range(tree.tree_.node_count)
            if tree.tree_.children_left[index] == -1}

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
            return self.tree.predict_proba(features)[0][1]

        # Else, return an array with label predictions
        return self.tree.predict_proba(features)[:, 1]

    def gradientize(
            self,
            features,
            epsilon=1e-12):
        """
        Calculate the tree gradient.

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

        def get_shift(bounds):
            """Get the projection on a bound set."""

            # Calculate the projection shift
            shift = features - clip(
                features, bounds[:, 0]+epsilon, bounds[:, 1]-epsilon)

            # Get the prediction value for the shifted features
            value = self.predict_proba(features-shift)

            return shift.reshape(-1), npnorm(shift), value

        # Get the prediction value
        prediction = self.predict_proba(features)

        # Get all leafs with lower prediction value
        targets = {
            key: self.bounds[key] for key, value in self.outcomes.items()
            if value < prediction}

        # Check if any "better" leafs have been found
        if len(targets) > 0:

            # Get the projection results
            projections = list(map(get_shift, targets.values()))

            # Get the projection with minimum norm
            shift, norm, value = min(projections, key=lambda x: x[1])

            # Return the gradient
            return -shift*(value-prediction)/norm**2

        # Else, return the zero gradient
        return zeros(features.shape[1])
