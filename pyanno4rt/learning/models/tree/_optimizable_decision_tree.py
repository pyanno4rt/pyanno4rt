"""Optimizable decision tree."""

# Author: Tim Ortkamp

# %% External package import

from operator import gt, itemgetter, le

from itertools import chain, groupby
from numpy import argmin, argwhere, array, prod, zeros
from numpy.linalg import norm

# %% Class definition


class OptimizableDecisionTree():
    """
    Optimizable decision tree class.

    This class implements an optimizable surrogate model for scikit-learn's \
    decision tree classifier. It exploits the pre-fitted structure of the \
    classifier to express the probability prediction function as a sum of \
    path-wise weighted products of indicator functions, and approximates an \
    input gradient using the minimum input feature shift required to improve \
    the prediction value.

    Attributes
    ----------
    paths : dict
        Dictionary with information on the decision tree paths.
    """

    def __init__(self):

        # Initialize the path dictionary as a list
        self.paths = []

    def traverse(
            self,
            tree):
        """
        Read the path information from the pre-fitted decision tree.

        Parameters
        ----------
        tree : object of class :class:`sklearn.tree.DecisionTreeClassifier`
            The object used to represent the pre-fitted decision tree.
        """

        # Get the structural properties of the tree object
        (node_count, children_left, children_right, feature, threshold,
         value) = (getattr(tree.tree_, name) for name in (
             'node_count', 'children_left', 'children_right', 'feature',
             'threshold', 'value'))

        # Initialize the storage lists
        nodes, thresholds, signs = [], [], []

        # Get the boolean leaf indicators
        is_leaf = [
            left_child == right_child
            for left_child, right_child in zip(children_left, children_right)]

        # Loop over the number of nodes
        for i in range(node_count):

            # Check if the current node is the root
            if i == 0:

                # Append the initial left and right path
                nodes.append([i, children_left[i]])
                nodes.append([i, children_right[i]])

                # Append the initial left and right thresholds
                thresholds.append(
                    [threshold[i], threshold[children_left[i]]])
                thresholds.append(
                    [threshold[i], threshold[children_right[i]]])

                # Append the initial left and right signs
                signs.append([le])
                signs.append([gt])

            # Check if the current node is a leaf
            elif is_leaf[i]:

                # Get the end nodes of all current paths
                end_nodes = array([node[-1] for node in nodes])

                # Check if the current node terminates a path
                if i in end_nodes:

                    # Get the feature numbers for the path
                    features = [
                        feature[node] for node in
                        nodes.pop(argwhere(i == end_nodes)[0][0])[:-1]]

                    # Construct the data tuple for the path
                    path_data = (
                        features,
                        thresholds.pop(argwhere(i == end_nodes)[0][0]),
                        signs.pop(argwhere(i == end_nodes)[0][0]),
                        value[i][:, 1][0])

                    # Append the data to the path list
                    self.paths.append(dict(zip(
                        ('nodes', 'thresholds', 'signs', 'value'), path_data)))

            else:

                # Get the left and right children of the current node
                left_children, right_children = (
                    children_left[i], children_right[i])

                # Loop over the storage lists
                for j, lists in enumerate(zip(nodes, thresholds, signs)):

                    # Check if the current node terminates the current path
                    if i == lists[0][-1]:

                        # Extend the paths by the children nodes
                        nodes[j] = lists[0] + [left_children]
                        nodes.append(lists[0] + [right_children])

                        # Extend the thresholds by the children thresholds
                        thresholds[j] = lists[1] + (
                            [threshold[left_children]]
                            * int(threshold[left_children] != -2.0))
                        thresholds.append(lists[1] + (
                            [threshold[right_children]]
                            * int(threshold[right_children] != -2.0)))

                        # Extend the signs by the children signs
                        signs[j] = lists[2] + [le]
                        signs.append(lists[2] + [gt])

        # Convert the path list into a structured path dictionary
        self.paths = {
            key: tuple(data) for (key, data) in groupby(
                sorted(self.paths, key=itemgetter('value'), reverse=True),
                itemgetter('value'))}

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

            # Calculate the prediction value
            prediction = sum(prod([path['signs'][i](
                features[0, path['nodes'][i]], path['thresholds'][i])
                for i in range(len(path['nodes']))])*path['value']
                for path in chain.from_iterable(self.paths.values()))

            # Append the prediction value to the list
            predictions.append([1-prediction, prediction])

        else:

            # Loop over the samples in the feature array
            for sample in features:

                # Calculate the prediction value
                prediction = sum(prod([path['signs'][i](
                    sample[path['nodes'][i]], path['thresholds'][i])
                    for i in range(len(path['nodes']))])*path['value']
                    for path in chain.from_iterable(self.paths.values()))

                # Append the prediction value to the list
                predictions.append([1-prediction, prediction])

        # Return the label predictions
        return array(predictions)

    def gradientize(
            self,
            features):
        """
        Gradientize the features with the minimum distance improvement shift.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        Returns
        -------
        ndarray
            Values of the gradient.
        """

        def calculate_shift(features, path):
            """Calculate the required feature shift towards a path."""

            # Initialize the shift vector
            shift = zeros(features.shape[1])

            # Get the path structure
            nodes, thresholds, signs = (
                path[key] for key in ('nodes', 'thresholds', 'signs'))

            # Loop over the path nodes
            for i, node in enumerate(nodes):

                # Check if the path condition is not fulfilled
                if not signs[i](features[0, node], thresholds[i]):

                    # Calculate the shift
                    shift[node] = (
                        thresholds[i] - features[0, node]
                        + 1e-12*(signs[i].__name__ == 'gt'))

            # Return the shift array and the l2-norm
            return shift, norm(shift)

        # Get the prediction value
        prediction = self.predict_proba(features)[0][1]

        # Check if the prediction value is not yet minimal
        if len(self.paths) > 0 and prediction != tuple(self.paths)[-1]:

            # Get the next best prediction values
            temp_list = list(self.paths)
            next_values = temp_list[temp_list.index(prediction)+1:]

            # Calculate the next best shifts
            shifts = [
                calculate_shift(features, path) for value in next_values
                for path in self.paths[value]]

            # Get the index of the minimum distance shift
            index = argmin(shift[1] for shift in shifts)

            # Get the minimum shift
            shift, length = shifts[index]

            # Return the gradient value
            return (shift*(next_values[index] - prediction)/length**2)

        # Otherwise, return the zero gradient
        return zeros(features.shape[1])
