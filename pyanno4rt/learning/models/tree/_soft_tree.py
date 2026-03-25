"""Soft decision tree."""

# Author: Tim Ortkamp
# Inspired by Irsoy et al. (2012):
# https://ieeexplore.ieee.org/document/6460506

# %% External package import

from numpy import (
    append, clip, column_stack, concatenate, full, mean, newaxis, ones, unique,
    zeros_like)
from numpy import any as nany
from numpy import sum as nsum
from numpy.random import default_rng
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# %% Internal package import

from pyanno4rt.learning.losses import focal_loss, log_loss
from pyanno4rt.tools import sigmoid

# %% Class definition


class Node():
    """
    Node class.

    This class implements a node for the soft decision tree.

    Parameters
    ----------
    parent : None or object of class \
        :class:`~pyanno4rt.learning.tree._soft_tree.Node`
        The object used to represent the parent node.

    outcome : int or float, default=0.5
        Outcome value at the node.

    Attributes
    ----------
    parent : None or object of class \
        :class:`~pyanno4rt.learning.tree._soft_tree.Node`
        See 'Parameters'.

    left_child : None or object of class \
        :class:`~pyanno4rt.learning.tree._soft_tree.Node`
        The object used to represent the left child of the node.

    right_child : None or object of class \
        :class:`~pyanno4rt.learning.tree._soft_tree.Node`
        The object used to represent the right child of the node.

    weight : None or ndarray
        Weight vector for the linear term in the sigmoid function.

    bias : None, int or float
        Bias for the linear term in the sigmoid function.

    outcome : None, int or float
        See 'Parameters'.
    """

    def __init__(
            self,
            parent=None,
            outcome=0.5):

        # Initialize the relative nodes
        self.parent = parent
        self.left_child = None
        self.right_child = None

        # Initialize the weight, bias and outcome value
        self.weight = None
        self.bias = None
        self.outcome = outcome

    def is_leaf(self):
        """Check if the node is a leaf."""

        return self.left_child is None and self.right_child is None

    def evaluate_subtree(
            self,
            X,
            temperature=1.0):
        """
        Recursively evaluate the subtree prediction(s).

        Parameters
        ----------
        X : ndarray
            Feature values.

        temperature : float, default=1.0
            Logit scaling parameter.

        Returns
        -------
        ndarray
            Subtree prediction(s).
        """

        # Check if the node is a leaf
        if self.is_leaf():

            # Return the leaf outcome probability
            return full(X.shape[0], self.outcome)

        # Calculate the routing probability
        prob = sigmoid((X @ self.weight + self.bias) / temperature)

        # Return the recursive prediction
        return (
            (1 - prob) * self.left_child.evaluate_subtree(X, temperature)
            + prob * self.right_child.evaluate_subtree(X, temperature))

    def evaluate_subtree_gradient(
            self,
            X,
            temperature=1.0):
        """
        Recursively evaluate the subtree gradient with respect to the input X.

        Parameters
        ----------
        X : ndarray
            Feature values.

        temperature : float, default=1.0
            Logit scaling parameter.

        Returns
        -------
        ndarray
            Subtree gradient.
        """

        # Check if the node is a leaf
        if self.is_leaf():

            # Return a zero gradient
            return zeros_like(X)

        # Calculate the routing probability
        prob = sigmoid((X @ self.weight + self.bias) / temperature)

        # Get the sigmoid function gradient
        sigmoid_grad = (prob * (1 - prob)) / temperature

        # Evaluate the left and right subtree
        left_out = self.left_child.evaluate_subtree(X, temperature)
        right_out = self.right_child.evaluate_subtree(X, temperature)

        # Calculate the gradient of the routing part
        routing_grad = (
            ((right_out - left_out)[:, newaxis] * sigmoid_grad[:, newaxis])
            * self.weight)

        # Recursively calculate the gradient of the child nodes
        children_grad = (
            (1-prob)[:, newaxis]
            * self.left_child.evaluate_subtree_gradient(X, temperature)
            + prob[:, newaxis]
            * self.right_child.evaluate_subtree_gradient(X, temperature))

        return routing_grad + children_grad


class SoftTree(BaseEstimator, ClassifierMixin):
    """
    Soft decision tree class.

    This class implements a soft decision tree with sigmoid test functions, \
    including methods to fit the model and make predictions.

    Parameters
    ----------
    criterion : {'focal_loss', 'log_loss'}, default='log_loss'
        Criterion for optimizing the subtree weights.

    max_depth : int, default=None
        Maximum depth of the tree.

    temperature : float, default=1.0
        Logit scaling parameter.

    tolerance : int or float, default=1e-3
        Precision goal for the criterion in each split.

    Attributes
    ----------
    criterion : {'focal_loss', 'log_loss'}
        See 'Parameters'.

    max_depth : int
        See 'Parameters'.

    temperature : float
        See 'Parameters'.

    tolerance : int or float
        See 'Parameters'.

    root : None or object of class \
        :class:`~pyanno4rt.learning.tree._soft_tree.Node`
        Root node of the soft decision tree.
    """

    def __init__(
            self,
            criterion='log_loss',
            max_depth=None,
            temperature=1.0,
            tolerance=1e-3):

        # Initialize the attributes
        self.criterion = criterion
        self.max_depth = max_depth or 2147483647
        self.temperature = temperature
        self.tolerance = tolerance

        # Initialize the root node
        self.root = None

        # Initialize the data information variables
        self.n_features_in_ = None
        self.classes_ = None

    def _get_loss_fn(self):
        """
        Get the loss function from the criterion.

        Returns
        -------
        callable : :func:`~pyanno4rt.learning.losses.log_loss` or \
            :func:`~pyanno4rt.learning.losses.focal_loss`
            Loss function.
        """

        # Map the string criterion to the loss functions
        mapping = {'focal_loss': focal_loss, 'log_loss': log_loss}

        return mapping.get(self.criterion, log_loss)

    def fit(
            self,
            X,
            y):
        """
        Fit the soft decision tree classifier.

        Parameters
        ----------
        X : ndarray
            Feature values.

        y : ndarray
            Label values.

        Returns
        -------
        self : object of class \
            :class:`~pyanno4rt.learning.tree._soft_tree.SoftTree`
            The object used to represent the soft decision tree.
        """

        def learn_subtree(node, features, labels, weights, depth):
            """Learn the subtree rooted at a node."""

            def objective(parameters):

                # Get the weight, bias and left/right outcome values
                weight = parameters[:self.n_features_in_]
                bias = parameters[self.n_features_in_]
                left_out, right_out = parameters[-2], parameters[-1]

                # Calculate the routing probability
                prob = sigmoid((features @ weight + bias) / self.temperature)

                # Get the weighted prediction
                prediction = (1 - prob) * left_out + prob * right_out

                # Clip the prediction
                predictions = clip(prediction, 1e-15, 1 - 1e-15)

                # Get the weight mask
                relevant_mask = weights > 1e-6

                # Check if no weights are greater than 1e-6
                if not nany(relevant_mask):

                    # Return the zero loss
                    return 0.0

                #
                relevant_labels = labels[relevant_mask]
                relevant_predictions = predictions[relevant_mask]

                #
                padded_labels = append(relevant_labels, [0, 1])
                padded_predictions = append(relevant_predictions, [0.5, 0.5])

                # Calculate the total loss
                total_loss = loss_fn(padded_labels, padded_predictions)

                # Return the weighted loss
                return total_loss * (nsum(weights) / len(weights))

            # Check the termination criteria
            if (depth >= self.max_depth
                    or nsum(weights) < 1e-5
                    or len(unique(labels)) < 2
                    or len(labels) < 2):

                return

            try:

                # Calculate the base loss
                base_loss = loss_fn(labels, full(len(labels), node.outcome))

            except ValueError:

                return

            # Initialize the random seed
            rng = default_rng(42)

            # Initialize the optimization variables
            initial_parameters = concatenate([
                rng.standard_normal(self.n_features_in_) * 0.1,  # weights
                [0.0],  # bias
                [max(0.01, node.outcome - 0.1),
                 min(0.99, node.outcome + 0.1)]  # initial outcomes
            ])

            # Initialize the variable bounds
            bounds = (
                [(None, None)] * (self.n_features_in_ + 1)
                + [(1e-7, 1-1e-7), (1e-7, 1-1e-7)])

            # Optimize the variables
            result = minimize(
                objective, initial_parameters, method='L-BFGS-B',
                bounds=bounds, tol=1e-3)

            # Compute the improvement
            improvement = base_loss - result.fun

            # Check if optimization has succeeded with sufficient improvement
            if result.success and improvement > self.tolerance:

                # Set the node attributes
                node.weight = result.x[:self.n_features_in_]
                node.bias = result.x[self.n_features_in_]
                node.left_child = Node(parent=node, outcome=result.x[-2])
                node.right_child = Node(parent=node, outcome=result.x[-1])

                # Calculate the routing probability
                prob = sigmoid(
                    (features @ node.weight + node.bias) / self.temperature)

                # Update the weights
                left_weights = weights * (1 - prob)
                right_weights = weights * prob

                # Recursively learn the left/right subtree
                learn_subtree(
                    node.left_child, features, labels, left_weights, depth+1)
                learn_subtree(
                    node.right_child, features, labels, right_weights, depth+1)

            else:

                # Reset the node to a leaf
                node.weight = None
                node.bias = None
                node.left_child = None
                node.right_child = None

        # Check the input
        X, y = check_X_y(X, y)

        # Extract data information
        self.n_features_in_ = X.shape[1]
        self.classes_ = unique(y)

        # Convert the labels to float
        y = y.astype(float)

        # Get the loss function
        loss_fn = self._get_loss_fn()

        # Initialize the weights
        initial_weights = ones(len(y))

        # Create the root node
        self.root = Node(outcome=mean(y))

        # Learn the subtree(s)
        learn_subtree(self.root, X, y, initial_weights, 0)

        return self

    def predict_proba(
            self,
            X):
        """
        Predict the probabilistic label value(s).

        Parameters
        ----------
        X : ndarray
            Feature values.

        Returns
        -------
        float or ndarray
            Predicted probabilistic label value(s).
        """

        # Check if the classifier has been fitted
        check_is_fitted(self)

        # Check the input array
        X = check_array(X)

        # Get the subtree probability
        probability = self.root.evaluate_subtree(X, self.temperature)

        # Return the probabilities for both classes
        return column_stack([1-probability, probability])

    def predict(
            self,
            X):
        """
        Predict the binary label value(s).

        Parameters
        ----------
        X : ndarray
            Feature values.

        Returns
        -------
        float or ndarray
            Predicted binary label value(s).
        """

        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def gradientize(
            self,
            X):
        """
        Calculate the gradient of the input features.

        Parameters
        ----------
        X : ndarray
            Feature values.

        Returns
        -------
        ndarray
            Gradient vector.
        """

        # Check if the classifier has been fitted
        check_is_fitted(self)

        # Check the input array
        X = check_array(X)

        return self.root.evaluate_subtree_gradient(X, self.temperature)
