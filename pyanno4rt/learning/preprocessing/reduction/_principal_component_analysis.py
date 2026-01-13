"""Principal component analysis."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from sklearn.decomposition import PCA as ScikitPCA

# %% Internal package import

from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_item, validate_item_in_set, validate_type)

# %% Class definition


class PrincipalComponentAnalysis():
    """
    Principal component analysis (PCA) class.

    This class provides methods to fit a principal component analyzer as well \
    as transform and gradientize input features.

    Parameters
    ----------
    n_components : int, float or 'mle', default=None
        Number of components to keep. For details, see the class \
        :class:`~sklearn.decomposition.PCA`.

    svd_solver : {'auto', 'full', 'covariance_eigh', 'arpack', 'randomized'}, \
        default='auto'
        Solver for the SVD. For details, see the class \
        :class:`~sklearn.decomposition.PCA`.

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

    model : object of class :class:`~sklearn.decomposition.PCA`
        The object used to represent the principal component analyzer.
    """

    # Set the algorithm name
    _name = 'PCA'

    # Set the skip indicator
    _train_only = False

    def __init__(
            self,
            n_components=None,
            svd_solver='auto'):

        # Get the input arguments
        self.arguments = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.validate(self.arguments)

        # Initialize the hyperparameters
        self.hyperparameters = {
            'n_components': n_components,
            'copy': True,
            'whiten': False,
            'svd_solver': svd_solver,
            'tol': 0.0,
            'iterated_power': 'auto',
            'n_oversamples': 10,
            'power_iteration_normalizer': 'auto',
            'random_state': None}

        # Initialize the PCA model
        self.model = ScikitPCA(**self.hyperparameters)

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
        Serialize the PCA model into a dictionary.

        Returns
        -------
        dict
            Dictionary with the model's arguments.
        """

        return {self._name: self.arguments}

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the PCA model from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the model's arguments.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.preprocessing.reduction._principal_component_analysis.PrincipalComponentAnalysis`
            The object used to represent the PCA model.
        """

        return cls(**dictionary)

    def fit(
            self,
            features,
            _):
        """
        Fit the PCA model.

        Parameters
        ----------
        features : ndarray
            Feature values.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.preprocessing.reduction._principal_component_analysis.PrincipalComponentAnalysis`
            Reference to the instance (self).
        """

        # Fit the PCA model
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

        return self.model.transform(features), labels

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

        return self.model.fit_transform(features, labels)

    def compute_gradient(
            self,
            _):
        """
        Compute the input gradient.

        Returns
        -------
        ndarray
            Input gradient.
        """

        return self.model.components_.T

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
            'n_components': (
                partial(validate_type, options=(int, float, str, type(None))),
                ),
            'svd_solver': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=(
                    'auto', 'full', 'covariance_eigh', 'arpack', 'randomized'))
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)

            # Check if the key is 'n_components'
            if key == 'n_components':

                # Check if the value is an integer
                if isinstance(value, int):

                    # Check if the value is at least 1
                    validate_item(key, value, reference=1, sign='>=')

                # Check if the value is a float
                elif isinstance(value, float):

                    # Check if the value is between 0 and 1
                    validate_item(key, value, reference=0, sign='>')
                    validate_item(key, value, reference=1, sign='<')

                # Check if the value is a string
                elif isinstance(value, str):

                    # Check if the value is 'mle'
                    validate_item_in_set(key, value, options=('mle',))
