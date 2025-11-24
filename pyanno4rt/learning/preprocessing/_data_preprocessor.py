"""Data preprocessing."""

# Author: Tim Ortkamp

# %% External package import

from numpy import prod, vstack

# %% Internal package import

import pyanno4rt.learning._maps as maps
from pyanno4rt.logging import get_logger

# %% Class definition


class DataPreprocessor():
    """
    Data preprocessing class.

    Parameters
    ----------
    sequence : list
        Labels for the preprocessing algorithm classes (sequential).

    verbose : bool, default=True
        Indicator for the display of logging messages.

    Attributes
    ----------
    steps : dict
        Dictionary with the preprocessing steps as pairs of labels and \
        corresponding preprocessing algorithm classes.
    """

    def __init__(
            self,
            sequence,
            verbose=True):

        # Check if a message should be printed
        if verbose:

            # Log a message about the initialization of the class
            get_logger().info("Initializing data preprocessor ...")

        # Merge the preprocessing algorithm maps
        preprocessing_map = {**maps.TRANSFORMERS}

        # Check if the sequence is empty
        if len(sequence) == 0:

            # Overwrite the sequence by the identity step
            sequence = ['Identity']

        # Check if a message should be printed
        if verbose:

            # Log a message about the pipeline build
            get_logger().info(
                "Building pipeline 'Input -> %s -> Output' ...",
                ' -> '.join(sequence))

        # Generate the preprocessing steps dictionary
        self.steps = dict(zip(
            sequence, (preprocessing_map[label]() for label in sequence)))

    def transform(
            self,
            features,
            labels=None):
        """
        Transform the input features/labels.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : ndarray, default=None
            Values of the input labels.

        Returns
        -------
        ndarray
            Transformed values of the input features.

        None or ndarray
            Transformed values of the input labels.
        """

        # Loop over the preprocessing steps
        for step in self.steps.values():

            # Transform the features and labels
            features, labels = step.transform(features, labels)

        return features, labels

    def fit_transform(
            self,
            features,
            labels=None):
        """
        Fit the preprocessor and transform the input features/labels.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : ndarray, default=None
            Values of the input labels.

        Returns
        -------
        ndarray
            Transformed values of the input features.

        None or ndarray
            Transformed values of the input labels.
        """

        # Loop over the preprocessing steps
        for step in self.steps.values():

            # Fit the algorithm
            step.fit(features, labels)

            # Transform the features and labels
            features, labels = step.transform(features, labels)

        return features, labels

    def gradientize(
            self,
            features):
        """
        Compute the preprocessing gradient w.r.t the input features.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        Returns
        -------
        ndarray
            Value of the preprocessing gradient.
        """

        return prod(vstack(tuple(
            step.compute_gradient(features) for step in self.steps.values()
            if hasattr(step, 'compute_gradient') and callable(
                    step.compute_gradient))), axis=0)
