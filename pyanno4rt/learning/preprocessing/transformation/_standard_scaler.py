"""Standard scaling."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from numpy import array, diag, mean, ones, std, zeros

# %% Internal package import

from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import validate_type

# %% Class definition


class StandardScaler():
    """
    Standard scaling class.

    This class provides methods to fit a standard scaler as well as \
    transform and gradientize input features.

    Parameters
    ----------
    center : bool, default=True
        Indicator for the mean centering of the data.

    scale : bool, default=True
        Indicator for the variance scaling of the data.

    Attributes
    ----------
    _name : str
        Name of the preprocessing algorithm.

    _train_only : bool
        Indicator for skipping the algorithm during prediction.

    arguments : dict
        Dictionary with the input arguments (for serialization).

    center : bool
        See 'Parameters'.

    scale : bool
        See 'Parameters'.

    means : None or ndarray
        Mean values of the features.

    deviations : None or ndarray
        Standard deviations of the features.
    """

    # Set the algorithm name
    _name = 'StandardScaler'

    # Set the skip indicator
    _train_only = False

    def __init__(
            self,
            center=True,
            scale=True):

        # Get the input arguments
        self.arguments = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.validate(self.arguments)

        # Get the instance attributes
        self.center = center
        self.scale = scale

        # Initialize the means and standard deviations
        self.means, self.deviations = None, None

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
        Serialize the standard scaler into a dictionary.

        Returns
        -------
        dict
            Dictionary with the scaler's arguments.
        """

        return {self._name: self.arguments}

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the standard scaler from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the scaler's arguments.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.preprocessing.transformation._standard_scaler.StandardScaler`
            The object used to represent the standard scaler.
        """

        return cls(**dictionary)

    def fit(
            self,
            features,
            _):
        """
        Fit the standard scaler.

        Parameters
        ----------
        features : ndarray
            Feature values.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.preprocessing.transformation._standard_scaler.StandardScaler`
            Reference to the instance (self).
        """

        # Check if the features should be centered
        if self.center:

            # Compute the mean values of the features
            self.means = mean(features, axis=0)

        else:

            # Set the means to the default
            self.means = zeros((features.shape[1],))

        # Check if the features should be scaled
        if self.scale:

            # Compute the standard deviations of the features
            self.deviations = std(features, axis=0)
            self.deviations[self.deviations == 0] = 1.0

        else:

            # Set the standard deviations to the default
            self.deviations = ones((features.shape[1],))

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

        return array(
           [(features[:, index]-self.means[index])/self.deviations[index]
            for index in range(features.shape[1])]).T, labels

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

    def compute_gradient(
            self,
            features):
        """
        Compute the input gradient.

        Parameters
        ----------
        features : ndarray
            Feature values.

        Returns
        -------
        ndarray
            Input gradient.
        """

        return diag(array(
            [1/self.deviations[index] for index in range(features.shape[1])]))

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
            'center': (
                partial(validate_type, options=bool),
                ),
            'scale': (
                partial(validate_type, options=bool),
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
