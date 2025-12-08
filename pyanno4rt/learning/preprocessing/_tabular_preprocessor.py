"""Tabular data preprocessing."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from numpy import prod, vstack

# %% Internal package import

import pyanno4rt.learning._maps as maps
from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_item_in_set, validate_subtype, validate_type)

# %% Class definition


class TabularPreprocessor():
    """
    Tabular data preprocessing class.

    Parameters
    ----------
    steps : list
        Labels for the preprocessing algorithms.

    Attributes
    ----------
    steps : list
        Preprocessing steps.
    """

    def __init__(
            self,
            steps):

        # Get the input arguments
        self.inputs = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.validate(self.inputs)

        # Get the preprocessing steps
        self.steps = [maps.TRANSFORMERS[label]() for label in steps]

    def to_dict(self):
        """Serialize the preprocessor into a dictionary."""

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """Deserialize the preprocessor parameters from a dictionary."""

    def fit(
            self,
            features,
            labels=None):
        """
        Fit the preprocessor.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : ndarray, default=None
            Values of the input labels.
        """

        # Loop over the preprocessing steps
        for step in self.steps:

            # Fit the algorithm
            step.fit(features, labels)

            # Transform the features and labels
            features, labels = step.transform(features, labels)

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
        for step in self.steps:

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
        for step in self.steps:

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
            step.compute_gradient(features) for step in self.steps
            if hasattr(step, 'compute_gradient') and callable(
                    step.compute_gradient))), axis=0)

    def validate(
            self,
            inputs):
        """
        Validate the input arguments.

        Parameters
        ----------
        inputs : dict
            Dictionary with the mappings between argument names and values.
        """

        validation_map = {
            'steps': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=str),
                partial(validate_item_in_set, options=tuple(maps.TRANSFORMERS))
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
