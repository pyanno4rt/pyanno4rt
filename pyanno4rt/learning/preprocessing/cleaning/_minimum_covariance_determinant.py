"""Minimum covariance determinant."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from sklearn.covariance import EllipticEnvelope

# %% Internal package import

from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import validate_item, validate_type

# %% Class definition


class MinimumCovarianceDeterminant():
    """
    Minimum covariance determinant (MCD) class.

    This class provides methods to fit a MCD detector as well as transform \
    input features and labels.

    Parameters
    ----------
    contamination : float, default=0.1
        Expected proportion of outliers. Range is (0, 0.5].

    Attributes
    ----------
    _name : str
        Name of the preprocessing algorithm.

    _kind : str
        Type of preprocessing algorithm.

    arguments : dict
        Dictionary with the input arguments (for serialization).

    hyperparameters : dict
        Dictionary with the model hyperparameters.

    model : object of class :class:`~sklearn.covariance.EllipticEnvelope`
        The object used to represent the MCD detector.
    """

    # Set the algorithm name
    _name = 'MCD'

    # Set the algorithm type
    _kind = 'outlier_removal'

    def __init__(
            self,
            contamination=0.1):

        # Get the input arguments
        self.arguments = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.validate(self.arguments)

        # Initialize the hyperparameters
        self.hyperparameters = {
            'store_precision': True,
            'assume_centered': False,
            'support_fraction': None,
            'contamination': contamination,
            'random_state': 15}

        # Initialize the MCD detector
        self.model = EllipticEnvelope(**self.hyperparameters)

    @property
    def name(self):
        """Get the algorithm name."""
        return self._name

    @property
    def kind(self):
        """Get the algorithm type."""
        return self._kind

    def to_dict(self):
        """
        Serialize the MCD detector into a dictionary.

        Returns
        -------
        dict
            Dictionary with the detector's arguments.
        """

        return {self._name: self.arguments}

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the MCD detector from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the detector's arguments.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.preprocessing.cleaning._minimum_covariance_determinant.MinimumCovarianceDeterminant`
            The object used to represent the MCD detector.
        """

        return cls(**dictionary)

    def fit(
            self,
            features,
            _):
        """
        Fit the MCD detector.

        Parameters
        ----------
        features : ndarray
            Feature values.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.preprocessing.cleaning._minimum_covariance_determinant.MinimumCovarianceDeterminant`
            Reference to the instance (self).
        """

        # Fit the MCD detector
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
            'contamination': (
                partial(validate_type, options=float),
                partial(validate_item, reference=0, sign='>'),
                partial(validate_item, reference=0.5, sign='<=')
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
