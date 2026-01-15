"""Tabular data preprocessing."""

# Author: Tim Ortkamp

# %% External package import

from copy import deepcopy
from functools import partial, reduce
from numpy import matmul

# %% Internal package import

import pyanno4rt.learning._maps as maps
from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import validate_subtype, validate_type

# %% Class definition


class TabularPreprocessor():
    """
    Tabular data preprocessing class.

    Parameters
    ----------
    pipeline : list
        The objects used to represent the steps in the preprocessing pipeline.

        Currently available:

        - :class:`~pyanno4rt.learning.preprocessing.cleaning._isolation_forest.IsolationForest`

        - :class:`~pyanno4rt.learning.preprocessing.cleaning._local_outlier_factor.LocalOutlierFactor`

        - :class:`~pyanno4rt.learning.preprocessing.cleaning._minimum_covariance_determinant.MinimumCovarianceDeterminant`

        - :class:`~pyanno4rt.learning.preprocessing.reduction._principal_component_analysis.PrincipalComponentAnalysis`

        - :class:`~pyanno4rt.learning.preprocessing.transformation._standard_scaler.StandardScaler`

        - :class:`~pyanno4rt.learning.preprocessing.transformation._whitening.Whitening`

    Attributes
    ----------
    arguments : dict
        Dictionary with the input arguments (for serialization).

    pipeline : list
        See 'Parameters'.
    """

    def __init__(
            self,
            pipeline):

        # Get the input arguments
        self.arguments = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.validate(self.arguments)

        # Get the preprocessing pipeline
        self.pipeline = pipeline

    def to_dict(self):
        """
        Serialize the preprocessor into a dictionary.

        Returns
        -------
        dict
            Dictionary with the preprocessor's arguments.
        """

        # Get the parameter dictionary
        dictionary = deepcopy(self.arguments)

        # Serialize the pipeline
        dictionary['pipeline'] = [
            item.to_dict() for item in dictionary['pipeline']]

        return dictionary

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the preprocessor from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the preprocessor's arguments.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.preprocessing._tabular_preprocessor.TabularPreprocessor`
            The object used to represent the preprocessor.
        """

        # Deserialize the pipeline
        dictionary['pipeline'] = [
            maps.PREPROCESS_STEPS[key].from_dict(value)
            for item in dictionary['pipeline']
            for key, value in item.items()]

        return cls(**dictionary)

    def fit(
            self,
            features,
            labels=None):
        """
        Fit the preprocessor.

        Parameters
        ----------
        features : ndarray
            Feature values.

        labels : ndarray, default=None
            Label values.
        """

        # Loop over the preprocessing steps
        for step in self.pipeline:

            # Fit the algorithm
            step.fit(features, labels)

            # Transform the features and labels
            features, labels = step.transform(features, labels)

    def transform(
            self,
            features,
            labels=None,
            mode='predict'):
        """
        Transform the features and labels.

        Parameters
        ----------
        features : ndarray
            Feature values.

        labels : ndarray, default=None
            Label values.

        mode : {'fit', 'predict'}, default='predict'
            Mode of preprocessing. If 'predict', train-only preprocessing \
            steps are skipped, else all steps are traversed.

        Returns
        -------
        ndarray
            Transformed feature values.

        None or ndarray
            Transformed label values.
        """

        # Loop over the preprocessing steps
        for step in (
                step for step in self.pipeline
                if not (mode == 'predict')*step.train_only):

            # Transform the features and labels
            features, labels = step.transform(features, labels)

        return features, labels

    def fit_transform(
            self,
            features,
            labels=None):
        """
        Fit the preprocessor and transform the features and labels.

        Parameters
        ----------
        features : ndarray
            Feature values.

        labels : ndarray, default=None
            Label values.

        Returns
        -------
        ndarray
            Transformed feature values.

        None or ndarray
            Transformed label values.
        """

        # Loop over the preprocessing steps
        for step in self.pipeline:

            # Fit the algorithm
            step.fit(features, labels)

            # Transform the features and labels
            features, labels = step.transform(features, labels)

        return features, labels

    def gradientize(
            self,
            features):
        """
        Calculate the preprocessing gradient w.r.t the features.

        Parameters
        ----------
        features : ndarray
            Feature values.

        Returns
        -------
        ndarray
            Preprocessing gradient w.r.t the features.
        """

        # Initialize the list of gradients
        gradients = []

        # Loop over the differentiable steps
        for step in (
                step for step in self.pipeline
                if hasattr(step, 'compute_gradient')
                and callable(step.compute_gradient)):

            # Append the step gradient
            gradients.append(step.compute_gradient(features))

            # Transform the feature values
            features, _ = step.transform(features, None)

        return reduce(matmul, gradients)

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
            'pipeline': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=(
                    *maps.PREPROCESS_STEPS.values(),))
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
