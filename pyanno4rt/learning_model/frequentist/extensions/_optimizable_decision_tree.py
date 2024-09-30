"""Optimizable decision tree."""

# %% External package import

from numpy import argwhere, array, prod, zeros
from operator import gt, le

# %% Class definition


class OptimizableDecisionTree():
    """
    Optimizable decision tree class.

    This class implements an optimizable surrogate model for scikit-learn's \
    decision tree classifier. It exploits the pre-fitted structure of a \
    decision tree to express the probability prediction function as a sum of \
    path-wise products of indicator functions, and provides both an exact and \
    approximate formulation. For the approximation, it utilizes scaled \
    versions of the sigmoid function to yield an input-differentiable version \
    of the decision tree which qualifies for gradient-based optimizers.

    Attributes
    ----------
    paths : list
        List of dictionaries with information on the decision tree paths.

    value_function : object of class :class:`jaxlib.xla_extension.PjitFunction`
        The (pre-compiled) object used to approximate the probability \
        prediction function along a single decision tree path.

    gradient_function : object of class :class:`function`
        The (pre-compiled) object used to approximate the input gradient \
        function along a single decision tree path.
    """

    def __init__(self):

        # Initialize the path list
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

        # Get the required properties from the tree object
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

        # 
        self.paths = sorted(self.paths, key=lambda d: d['value'], reverse=True)

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

        # 
        predictions = []

        # Check if the feature array has only a single row
        if features.shape[0] == 1:

            # 
            prediction = sum(prod([path['signs'][i](
                features[0, path['nodes'][i]], path['thresholds'][i])
                for i in range(len(path['nodes']))])*path['value']
                for path in self.paths)

            # Return a single exact label prediction value
            predictions.append([1-prediction, prediction])

        else:

            for sample in features:

                # 
                prediction = sum(prod([path['signs'][i](
                    sample[path['nodes'][i]], path['thresholds'][i])
                    for i in range(len(path['nodes']))])*path['value']
                    for path in self.paths)

                predictions.append([1-prediction, prediction])

        # Otherwise, return an array with exact label predictions
        return array(predictions)

    def gradientize(
            self,
            features):
        """."""

        # Get the index of the current path
        index = next((index for (index, d) in enumerate(self.paths)
                      if d['value'] == self.predict_proba(features)[0][1]),
                     None)

        # 
        if index < len(self.paths)-1:

            shift = zeros(features.shape[1])
            eps = 1e-12

            nodes = self.paths[index+1]['nodes']
            thresholds = self.paths[index+1]['thresholds']
            signs = self.paths[index+1]['signs']

            for i, sample in enumerate(features):
                for j, node in enumerate(nodes):
                    if not signs[j](sample[node], thresholds[j]):
                        if signs[j] == le:
                            shift[node] = sample[node] - thresholds[j]
                        else:
                            shift[node] = sample[node] - thresholds[j] - eps

            return shift

        # 
        return zeros(features.shape[1])
