"""Whitening transformer."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import diag, mean
from numpy.linalg import eig

# %% Class definition


class Whitening():
    """
    Whitening transformer class.

    This class provides methods to fit, transform and gradientize the input \
    features by their whitening matrix.

    Parameters
    ----------
    method : {'pca', 'zca'}, default='zca'
        Method for the computation of the whitening matrix.

        - 'zca' : zero-phase component analysis (Mahalanobis transformation)
        - 'pca' : principal component analysis

    Attributes
    ----------
    method : {'pca', 'zca'}
        See 'Parameters'.

    means : ndarray
        Mean values of the features.

    matrix : ndarray
        Whitening matrix.
    """

    def __init__(
            self,
            method='zca'):

        # Get the whitening method from the arguments
        self.method = method

        # Initialize the means and the whitening matrix
        self.means = None
        self.matrix = None

    def fit(
            self,
            features,
            labels):
        """
        Fit the whitening transformer.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : None or ndarray
            Values of the input labels.
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
        covariance_matrix = ((centered_features.T @ centered_features)
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
            labels):
        """
        Transform the input features/labels.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : None or ndarray
            Values of the input labels.

        Returns
        -------
        ndarray
            Transformed values of the input features.

        None or ndarray
            Transformed values of the input labels.
        """

        return (features-self.means) @ self.matrix.T, labels

    def compute_gradient(
            self,
            features):
        """
        Compute the whitening transformer gradient w.r.t the input features.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        Returns
        -------
        ndarray
            Value of the whitening transformer gradient.
        """

        return self.matrix.mean(axis=1)
