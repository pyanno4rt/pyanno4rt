"""Optimizable decision tree."""

# %% External package import

from itertools import chain, groupby
from numpy import argwhere, array, prod, zeros
from numpy.linalg import norm
from operator import gt, itemgetter, le

# %% Class definition


class OptimizableDecisionTree():
    """
    Optimizable decision tree class.

    This class implements an optimizable surrogate model for scikit-learn's \
    decision tree classifier. It exploits the pre-fitted structure of a \
    decision tree to express the probability prediction function as a sum of \
    path-wise products of indicator functions, and approximates a decision \
    tree gradient as the minimum input feature shift required to improve the \
    prediction value.

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

        # Get the structure of the tree object
        (node_count, children_left, children_right, feature, threshold,
         value) = (getattr(tree.tree_, name) for name in (
             'node_count', 'children_left', 'children_right', 'feature',
             'threshold', 'value'))

        # Initialize the storage lists
        nodes, thresholds, signs = [], [], []

        # Get the boolean leaf indicators
        is_leaf = [cl == cr for cl, cr in zip(children_left, children_right)]

        # Loop over the number of nodes
        for i in range(node_count):

            # Check if the starting node is the tree root
            if i == 0:

                # Append the initial left/right path
                nodes.append([i, children_left[i]])
                nodes.append([i, children_right[i]])

                # Append the initial left/right thresholds
                thresholds.append(
                    [threshold[i], threshold[children_left[i]]])
                thresholds.append(
                    [threshold[i], threshold[children_right[i]]])

                # Append the initial left/right signs
                signs.append([le])
                signs.append([gt])

            # Check if the current node is a leaf
            elif is_leaf[i]:

                # Get the end nodes for all current paths
                end_node = array([node[-1] for node in nodes])

                # Check if the current node terminates a path
                if i in end_node:

                    # Get the feature numbers for the path
                    features = [
                        feature[node] for node in
                        nodes.pop(argwhere(i == end_node)[0][0])[:-1]]

                    # Construct the output tuple for the path
                    output = (
                        features,
                        thresholds.pop(argwhere(i == end_node)[0][0]),
                        signs.pop(argwhere(i == end_node)[0][0]),
                        value[i][:, 1][0])

                    # Append the converted output to the path list
                    self.paths.append(dict(zip(
                        ('nodes', 'thresholds', 'signs', 'value'), output)))

            else:

                # Get the left/right children of the current node
                cl, cr = children_left[i], children_right[i]

                # Loop over the current storage lists
                for j, lists in enumerate(zip(nodes, thresholds, signs)):

                    # Check if the current node terminates the current path
                    if i == lists[0][-1]:

                        # Extend the paths by the children nodes
                        nodes[j] = lists[0] + [cl]
                        nodes.append(lists[0] + [cr])

                        # Extend the thresholds by the children thresholds
                        thresholds[j] = lists[1] + (
                            [threshold[cl]]*int(threshold[cl] != -2.0))
                        thresholds.append(lists[1] + (
                            [threshold[cr]]*int(threshold[cr] != -2.0)))

                        # Extend the signs by the children signs
                        signs[j] = lists[2] + [le]
                        signs.append(lists[2] + [gt])

        # Convert the path list into a sorted/grouped path dictionary
        self.paths = {
            key: tuple(data) for (key, data) in
            groupby(sorted(self.paths, key=itemgetter('value'), reverse=True),
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
        Gradientize the features with the minimum improvement shift.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        Returns
        -------
        ndarray
            Values of the minimum improvement shift.
        """

        def calculate_shift(features, path):
            """Calculate the required feature shift towards a path."""

            # Initialize the shift vector
            shift = zeros(features.shape[1])

            # Initialize the perturbation variable
            eps = 1e-12

            # Get the path structure
            nodes, thresholds, signs = (
                path[key] for key in ('nodes', 'thresholds', 'signs'))

            # Loop over the path nodes
            for j, node in enumerate(nodes):

                # Check if the path condition is not fulfilled
                if not signs[j](features[0, node], thresholds[j]):

                    # Check if the condition sign is "<="
                    if signs[j] == le:

                        # Calculate the shift
                        shift[node] = features[0, node] - thresholds[j]

                    else:

                        # Otherwise, calculate the shift with perturbation
                        shift[node] = features[0, node] - thresholds[j] - eps

            # Return the shift array and its l2-norm
            return shift, norm(shift)

        # Get the current prediction value
        current_value = self.predict_proba(features)[0][1]

        # Check if the prediction value is not yet minimal
        if len(self.paths) > 0 and current_value != tuple(self.paths)[-1]:

            # Get the next best prediction values
            temp_list = list(self.paths)
            next_values = temp_list[temp_list.index(current_value)+1:]

            # Calculate next best shifts
            shifts = [calculate_shift(features, path)
                      for value in next_values
                      for path in self.paths[value]]

            # Return the minimum improvement shift
            return min(shifts, key=lambda shifts: shifts[1])[0]

        # Otherwise, return the zero-improvement shift
        return zeros(features.shape[1])
