"""Isolation forest."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from sklearn.ensemble import IsolationForest as ScikitIsoForest

# %% Internal package import

from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import validate_item, validate_type

# %% Class definition


class IsolationForest():
    """
    Isolation forest class.

    This class provides methods to fit an isolation forest as well as \
    transform input features and labels.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of base estimators.

    Attributes
    ----------
    _name : str
        Name of the preprocessing algorithm.

    _train_only : bool
        Indicator for skipping the algorithm during prediction.

    arguments : dict
        Dictionary with the input arguments (for serialization).

    hyperparameters : dict
        Dictionary with the model hyperparameters.

    model : object of class :class:`~sklearn.ensemble.IsolationForest`
        The object used to represent the isolation forest.
    """

    # Set the algorithm name
    _name = 'IsolationForest'

    # Set the skip indicator
    _train_only = True

    def __init__(
            self,
            n_estimators=100):

        # Get the input arguments
        self.arguments = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.validate(self.arguments)

        # Initialize the hyperparameters
        self.hyperparameters = {
            'n_estimators': n_estimators,
            'max_samples': 'auto',
            'contamination': 'auto',
            'max_features': 1.0,
            'bootstrap': False,
            'n_jobs': None,
            'random_state': 14,
            'verbose': 0,
            'warm_start': False}

        # Initialize the isolation forest
        self.model = ScikitIsoForest(**self.hyperparameters)

    @property
    def name(self):
        """Get the algorithm name."""
        return self._name

    @property
    def train_only(self):
        """Get the skip indicator."""
        return self._train_only

    def to_dict(self):
        """
        Serialize the isolation forest into a dictionary.

        Returns
        -------
        dict
            Dictionary with the forest's arguments.
        """

        return {self._name: self.arguments}

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the isolation forest from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the forest's arguments.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.preprocessing.cleaning._isolation_forest.IsolationForest`
            The object used to represent the isolation forest.
        """

        return cls(**dictionary)

    def fit(
            self,
            features,
            _):
        """
        Fit the isolation forest.

        Parameters
        ----------
        features : ndarray
            Feature values.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.preprocessing.cleaning._isolation_forest.IsolationForest`
            Reference to the instance (self).
        """

        # Fit the isolation forest
        self.model.fit(features)

        return self

    def transform(
            self,
            features,
            labels=None):
        """
        Transform the features and labels.

        Parameters
        ----------
        features : ndarray
            Feature values.

        labels : None or ndarray, default=None
            Label values.

        Returns
        -------
        ndarray
            Transformed feature values.

        None or ndarray
            (Transformed) label values.
        """

        # Get the inliers (=1) and outliers (=-1)
        predictions = self.model.predict(features)

        # Get the filter mask
        mask = predictions == 1

        # Filter the features
        features = features[mask]

        # Check if label values are provided
        if labels is not None:

            # Filter the labels
            labels = labels[mask]

        return features, labels

    def fit_transform(
            self,
            features,
            labels=None):
        """
        Fit and transform the features and labels.

        Parameters
        ----------
        features : ndarray
            Feature values.

        labels : None or ndarray, default=None
            Label values.

        Returns
        -------
        ndarray
            Transformed feature values.

        None or ndarray
            (Transformed) label values.
        """

        return self.fit(features, labels).transform(features, labels)

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
            'n_estimators': (
                partial(validate_type, options=int),
                partial(validate_item, reference=1, sign='>=')
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
