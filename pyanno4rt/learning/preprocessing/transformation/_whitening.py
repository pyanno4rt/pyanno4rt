"""Whitening."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from numpy import diag, mean
from numpy.linalg import eig

# %% Internal package import

from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import validate_item_in_set, validate_type

# %% Class definition


class Whitening():
    """
    Whitening class.

    This class provides methods to fit a whitening matrix as well as \
    transform and gradientize input features.

    Parameters
    ----------
    method : {'pca', 'zca'}, default='zca'
        Method for the computation of the whitening matrix.

        - 'zca' : zero-phase component analysis (Mahalanobis transformation)
        - 'pca' : principal component analysis

    Attributes
    ----------
    _name : str
        Name of the preprocessing algorithm.

    _kind : str
        Type of preprocessing algorithm.

    arguments : dict
        Dictionary with the input arguments (for serialization).

    method : {'pca', 'zca'}
        See 'Parameters'.

    means : None or ndarray
        Mean feature values.

    matrix : None or ndarray
        Whitening matrix.
    """

    # Set the algorithm name
    _name = 'Whitening'

    # Set the algorithm type
    _kind = 'scaling'

    def __init__(
            self,
            method='zca'):

        # Get the input arguments
        self.arguments = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.validate(self.arguments)

        # Get the calculation method
        self.method = method

        # Initialize the means and the whitening matrix
        self.means, self.matrix = None, None

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
        Serialize the whitening scaler into a dictionary.

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
        Deserialize the whitening scaler from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the scaler's arguments.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.preprocessing.transformation._whitening.Whitening`
            The object used to represent the whitening scaler.
        """

        return cls(**dictionary)

    def fit(
            self,
            features,
            _):
        """
        Fit the whitening matrix.

        Parameters
        ----------
        features : ndarray
            Feature values.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.preprocessing.transformation._whitening.Whitening`
            Reference to the instance (self).
        """

        def compute_zca_matrix(inverse_diagonal, eigenvectors):
            """Compute the whitening matrix from ZCA."""

            return inverse_diagonal @ eigenvectors.T

        def compute_pca_matrix(inverse_diagonal, eigenvectors):
            """Compute the whitening matrix from PCA."""

            return eigenvectors @ inverse_diagonal @ eigenvectors.T

        # Compute the mean features
        self.means = mean(features, axis=0)

        # Center the feature values
        centered_features = features - self.means

        # Compute the covariance matrix
        covariance_matrix = ((
            centered_features.T @ centered_features)
            / centered_features.shape[0])

        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eig(covariance_matrix)

        # Get the diagonal of the eigenvalues
        diagonal = diag(eigenvalues)

        # Compute the inverse-rooted diagonal
        inverse_diagonal = diag(diag(diagonal)**(-0.5))

        # Map the methods to the computation functions
        methods = {'zca': compute_zca_matrix, 'pca': compute_pca_matrix}

        # Compute the whitening matrix
        self.matrix = methods[self.method](inverse_diagonal, eigenvectors)

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

        return (features-self.means) @ self.matrix.T, labels

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
            _):
        """
        Compute the input gradient.

        Returns
        -------
        ndarray
            Input gradient.
        """

        return diag(self.matrix.mean(axis=1))

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
            'method': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=('pca', 'zca'))
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
