"""Whitening."""

# Author: Tim Ortkamp

# %% External package import

from numpy import diag, mean
from numpy.linalg import eig

# %% Class definition


class Whitening():
    """
    Whitening class.

    This class provides methods to fit a whitening matrix, transform and \
    gradientize the input features.

    Parameters
    ----------
    method : {'pca', 'zca'}, default='zca'
        Method for the computation of the whitening matrix.

        - 'zca' : zero-phase component analysis (Mahalanobis transformation)
        - 'pca' : principal component analysis

    Attributes
    ----------
    _classifier : str
        String indicating the preprocessing class.

    method : {'pca', 'zca'}
        See 'Parameters'.

    means : ndarray
        Mean values of the features.

    matrix : ndarray
        Whitening matrix.
    """

    # Initialize the algorithm classifier
    _classifier = 'scaling'

    def __init__(
            self,
            method='zca'):

        # Get the instance attributes from the arguments
        self.method = method

        # Initialize the means and the whitening matrix
        self.means = None
        self.matrix = None

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
        """

        def compute_zca_matrix(inverse_diagonal, eigenvectors):
            """Compute the whitening matrix from ZCA."""

            return inverse_diagonal @ eigenvectors.T

        def compute_pca_matrix(inverse_diagonal, eigenvectors):
            """Compute the whitening matrix from PCA."""

            return eigenvectors @ inverse_diagonal @ eigenvectors.T

        # Compute the means of the features
        self.means = mean(features, axis=0)

        # Center the feature values by the means
        centered_features = features-self.means

        # Compute the covariance matrix
        covariance_matrix = ((
            centered_features.T @ centered_features)
            / centered_features.shape[0])

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = eig(covariance_matrix)

        # Get the diagonal of the eigenvalues
        diagonal = diag(eigenvalues)

        # Compute the inverse-rooted diagonal
        inverse_diagonal = diag(diag(diagonal)**(-0.5))

        # Create a mapping between methods and computation functions
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
