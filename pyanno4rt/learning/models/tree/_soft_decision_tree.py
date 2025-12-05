"""Soft decision tree."""

# Author: Tim Ortkamp
# Inspired by Irsoy et al. (2012):
# https://ieeexplore.ieee.org/document/6460506

# %% External package import

from numpy import array, mean
from scipy.optimize import shgo

# %% Internal package import

from pyanno4rt.learning.losses import log_loss
from pyanno4rt.tools import sigmoid

# %% Class definition


class Node():
    """
    Soft decision tree node class.

    This class implements a decision node for the soft decision tree.

    Parameters
    ----------
    left_child : object of class \
        :class:`~pyanno4rt.learning.tree._soft_decision_tree.Node`
        The object used to represent the left child.

    right_child : object of class \
        :class:`~pyanno4rt.learning.tree._soft_decision_tree.Node`
        The object used to represent the right child.

    parent : object of class \
        :class:`~pyanno4rt.learning.tree._soft_decision_tree.Node`
        The object used to represent the parent node.

    Attributes
    ----------
    left_child : object of class \
        :class:`~pyanno4rt.learning.tree._soft_decision_tree.Node`
        See 'Parameters'.

    right_child : object of class \
        :class:`~pyanno4rt.learning.tree._soft_decision_tree.Node`
        See 'Parameters'.

    parent : object of class \
        :class:`~pyanno4rt.learning.tree._soft_decision_tree.Node`
        See 'Parameters'.

    weight : None, tuple, list or ndarray, default=None
        Weight vector for the linear term in the sigmoid function.

    outcome : None, int or float, default=None
        Outcome value at the node.
    """

    def __init__(
            self,
            left_child=None,
            right_child=None,
            parent=None):

        # Get the attributes from the arguments
        self.left_child = left_child
        self.right_child = right_child
        self.parent = parent

        # Initialize the weight vector and outcome value
        self.weight = None
        self.outcome = None

    def has_children(self):
        """
        Check if the node has children.

        Returns
        -------
        bool
            Indicator for the parenthood of the node.
        """

        return all(child is not None for child in (
            self.left_child, self.right_child))

    def set_weight(
            self,
            weight):
        """
        Set the weight vector for the linear term.

        Parameter
        ---------
        weight : tuple, list or ndarray
            Weight vector.
        """

        # Set the weight
        self.weight = weight

    def set_outcome(
            self,
            outcome):
        """
        Set the outcome for the node.

        Parameters
        ----------
        outcome : int or float
            Outcome value at the node.
        """

        # Set the outcome
        self.outcome = outcome

    def get_probability(
            self,
            sample):
        """
        Get the sigmoid function value for the input sample.

        Parameters
        ----------
        sample : ndarray
            Input sample.

        Returns
        -------
        float
            Sigmoid function value.
        """

        # Check if the weight vector has the correct length
        if self.weight is None or len(self.weight) != len(sample) + 1:

            # Raise an error to indicate incorrect weight or sample values
            raise ValueError(
                "Invalid weight vector or input sample - please check values \
                and/or size!")

        # Return the sigmoid function value
        return sigmoid(self.weight[1:]@sample, 1, self.weight[0])

    def evaluate_subtree(
            self,
            sample):
        """
        Calculate the outcome of the subtree for the input sample.

        Parameters
        ----------
        sample : ndarray
            Input sample.

        Returns
        -------
        float
            Outcome value.
        """

        # Check if the node is a leaf
        if not self.has_children():

            # Return the outcome value
            return self.outcome

        else:

            # Compute the test function
            probability = self.get_probability(sample)

            # Get the left and right subtree outcomes
            left_outcome = self.left_child.evaluate_subtree(sample)
            right_outcome = self.right_child.evaluate_subtree(sample)

            # Update the outcome
            return left_outcome*probability + right_outcome*(1-probability)

    def evaluate_subtree_gradient(
            self,
            sample):
        """
        Calculate the gradient of the subtree for the input sample.

        Parameters
        ----------
        sample : ndarray
            Input sample.

        Returns
        -------
        ndarray
            Gradient vector.
        """

        # Check if the node is a leaf
        if not self.has_children():

            # Return the outcome value
            return self.outcome

        else:

            # Compute the test function
            probability = self.get_probability(sample)

            # Get the left and right subtree gradients
            left_gradient = self.left_child.evaluate_subtree_gradient(sample)
            right_gradient = self.right_child.evaluate_subtree_gradient(sample)

            # Update the gradient
            return (
                left_gradient*probability*(1-probability)*self.weight
                - right_gradient*probability*(1-probability)*self.weight)


class SoftDecisionTree():
    """
    Soft decision tree class.

    This class implements a soft decision tree with sigmoid test functions, \
    including methods to fit the model and make predictions.

    Parameters
    ----------
    maximum_depth : int, default=None
        Maximum depth of the tree.

    tolerance : int or float, default=1e-3
        Precision goal for the loss function value in each split.

    Attributes
    ------
    maximum_depth : int
        See 'Parameters'.

    tolerance : int or float
        See 'Parameters'.

    root : object of class \
        :class:`~pyanno4rt.learning.tree._soft_decision_tree.Node`
        Root node of the soft decision tree.

    depth : int
        Depth of the soft decision tree.
    """

    def __init__(
            self,
            maximum_depth=None,
            tolerance=1e-3):

        # Get the attributes from the arguments
        self.maximum_depth = (
            2147483647 if maximum_depth is None else maximum_depth)
        self.tolerance = tolerance

        # Initialize the root node
        self.root = Node()

    def fit(
            self,
            features,
            labels):
        """
        Fit the soft decision tree classifier.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : ndarray
            Values of the input labels.

        Returns
        -------
        self : object of class \
            :class:`~pyanno4rt.learning.tree._soft_decision_tree.SoftDecisionTree`
            The object used to represent the soft decision tree.
        """

        def learn_subtree(node, features):
            """Learn the subtree rooted at a node."""

            def loss(parameters):
                """Return the loss function value w.r.t the parameters."""

                # Adapt the weight vector of the node
                node.set_weight(parameters[:-2])

                # Adapt the child outcomes of the node
                node.left_child.set_outcome(parameters[-2])
                node.right_child.set_outcome(parameters[-1])

                # Return the metric
                return log_loss(labels, self.predict_proba(features))

            # Increment the current depth number
            self.counter += 1

            # Get the initial loss
            initial_loss = log_loss(labels, self.predict_proba(features))

            # Initialize left and right child
            node.left_child = Node(parent=node)
            node.right_child = Node(parent=node)

            # Set the variable bounds
            bounds = tuple(
                (-5e0, 5e0) if i < features.shape[1]+1 else (0, 1)
                for i in range(features.shape[1]+3))

            # Optimize the weights and outcomes
            result = shgo(
                loss, bounds, n=100, iters=1, sampling_method='sobol')

            # Update the node parameters
            node.set_weight(result.x[:-2])
            node.left_child.set_outcome(result.x[-2])
            node.right_child.set_outcome(result.x[-1])

            # Get the final loss
            final_loss = log_loss(labels, self.predict_proba(features))

            # Check if the loss has improved from adding the subtree
            if (abs(final_loss - initial_loss) > self.tolerance
                    and self.counter <= 2**(self.maximum_depth)-1):

                # Learn the left and right subtree
                learn_subtree(node.left_child, features)
                learn_subtree(node.right_child, features)

            else:

                # Remove the child nodes
                node.left_child = None
                node.right_child = None

        # Set the mean label value as the default
        self.root.set_outcome(mean(labels))

        # Initialize the current depth number
        self.counter = 0

        # Learn the subtree
        learn_subtree(self.root, features)

        return self

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

        # Check if the feature array has only a single dimension
        if features.ndim == 1:

            # Return a single label prediction value
            return self.root.evaluate_subtree(features)

        # Else, return an array with label predictions
        return array([
            self.root.evaluate_subtree(sample) for sample in features])

    def gradientize(
            self,
            features):
        """
        Calculate the input gradient of the input sample X.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        Returns
        -------
        ndarray
            Gradient vector.
        """

        # Check if the feature array has only a single dimension
        if features.ndim == 1:

            # Return a single label prediction value
            return self.root.evaluate_subtree_gradient(features)[1:]

        # Else, return an array with label predictions
        return [
            self.root.evaluate_subtree_gradient(sample)[1:]
            for sample in features]

# %% Test

# from pandas import DataFrame
# from sklearn.datasets import load_iris
# from sklearn.metrics import roc_auc_score
# from sklearn.tree import DecisionTreeClassifier

# X, y = load_iris(return_X_y=True)
# X = X[:, [1, 3]]
# y = (y == 1).astype(int)
# soft_tree = SoftDecisionTree(maximum_depth=3, tolerance=1e-3).fit(X, y)
# hard_tree = DecisionTreeClassifier(max_depth=3).fit(X, y)

# df = DataFrame(
#     data=zip(
#         (hard_tree.predict_proba(sample.reshape(1, -1))[0][1] for sample in X),
#         soft_tree.predict_proba(X), y),
#     columns=['Hard', 'Soft', 'Ground Truth'])
# hard_auc = roc_auc_score(df['Ground Truth'], df['Hard'])
# soft_auc = roc_auc_score(df['Ground Truth'], df['Soft'])

# %% Plot

# from numpy import arange, meshgrid, array, ravel
# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x = y = arange(
#     min(min(X[:, 0]), min(X[:, 1])), max(max(X[:, 0]), max(X[:, 1])), 0.04)
# x1, x2 = meshgrid(x, y)

# zs1 = array([hard_tree.predict_proba(array([[a, b]]))[0][1]
#              for a, b in zip(ravel(x1), ravel(x2))])
# Z1 = zs1.reshape(x1.shape)

# zs2 = array([soft_tree.predict_proba(array([[a, b]]))
#              for a, b in zip(ravel(x1), ravel(x2))])
# Z2 = zs2.reshape(x1.shape)

# ax.plot_surface(x1, x2, Z1)
# ax.plot_surface(x1, x2, Z2)

# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_zlabel('p')

# plt.show()
