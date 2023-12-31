"""Whitening transformer."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import diag, mean
from numpy.linalg import eig
from sklearn.base import BaseEstimator, TransformerMixin

# %% Class definition


class Whitening(BaseEstimator, TransformerMixin):
    """
    Whitening transformer class.

    This class provides methods to fit the whitening matrix, transform input \
    features, and return the whitening gradient.

    Parameters
    ----------
    method : {'pca', 'zca'}, default = 'zca'
        Method for the computation of the whitening matrix. With 'zca', the \
        zero-phase component analysis (or Mahalanobis transformation) is \
        applied, with 'pca', the principal component analysis lays the \
        groundwork.

    Attributes
    ----------
    method : {'pca', 'zca'}
        See 'Parameters'.

    means : ndarray
        Mean values of the feature columns.

    matrix : ndarray
        Whitening matrix for the transformation of feature vectors.
    """

    # Set the algorithm label
    label = 'Whitening'

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
            labels=None):
        """
        Fit the transformator with the input data.

        Parameters
        ----------
        features : ndarray
        Values of the input features.

        labels : ndarray, default = None
            Values of the input labels.
        """

        def compute_zca_matrix(inverse_diagonal, eigenvectors):
            """Compute the whitening matrix from ZCA."""
            # Compute the whitening matrix
            self.matrix = inverse_diagonal @ eigenvectors.T

        def compute_pca_matrix(inverse_diagonal, eigenvectors):
            """Compute the whitening matrix from PCA."""
            # Compute the whitening matrix
            self.matrix = eigenvectors @ inverse_diagonal @ eigenvectors.T

        # Compute the means of the feature columns
        self.means = mean(features, axis=0)

        # Center the feature values by the column means
        centered_features = features - self.means

        # Compute the covariance matrix
        covariance_matrix = (
            (centered_features.T @ centered_features)
            / centered_features.shape[0])

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = eig(covariance_matrix)

        # Extract the diagonal of the eigenvalues
        diagonal = diag(eigenvalues)

        # Compute the inverse-rooted diagonal
        inverse_diagonal = diag(diag(diagonal)**(-0.5))

        # Map the methods with their computation functions
        methods = {'zca': compute_zca_matrix,
                   'pca': compute_pca_matrix}

        # Compute the whitening matrix
        methods[self.method](inverse_diagonal, eigenvectors)

        return self

    def transform(
            self,
            features,
            labels=None):
        """
        Transform the input data.

        Parameters
        ----------
        features : ndarray
        Values of the input features.

        labels : ndarray, default = None
            Values of the input labels.

        Returns
        -------
        ndarray
            Values of the transformed features.
        """
        return (features-self.means) @ self.matrix.T

    def compute_gradient(
            self,
            features):
        """
        Compute the gradient of the whitening transformation with respect to \
        the ``features``.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        Returns
        -------
        ndarray
            Values of the input feature gradients.
        """
        return self.matrix.mean(axis=1)
