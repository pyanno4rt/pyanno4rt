"""Standard scaling transformer."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import array, mean, ones, std, zeros
from sklearn.base import BaseEstimator, TransformerMixin

# %% Class definition


class StandardScaler(BaseEstimator, TransformerMixin):
    """
    Standard scaling transformer class.

    This class provides methods to fit the standard scaler, transform input \
    features, and return the scaler gradient.

    Parameters
    ----------
    center : bool
        Indicator for the computation of the mean values and centering of the \
        data.

    scale : bool
        Indicator for the computation of the standard deviations and the \
        scaling of the data.

    Attributes
    ----------
    center : bool
        See 'Parameters'.

    scale : bool
        See 'Parameters'.

    means : ndarray
        Mean values of the feature columns. Only computed if ``center``is \
        set to True, otherwise it is set to zeros.

    deviations : ndarray
        Standard deviations of the feature columns. Only computed if \
        ``scale``is set to True, otherwise it is set to ones
    """

    # Set the algorithm label
    label = 'StandardScaler'

    def __init__(
            self,
            center=True,
            scale=True):

        # Get the instance attributes from the arguments
        self.center = center
        self.scale = scale

        # Initialize the attributes for the means and standard deviations
        self.means = None
        self.deviations = None

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
        # Check if centering should be performed
        if self.center:

            # Compute the means of the feature columns
            self.means = mean(features, axis=0)

        else:

            # Set all means to zero (no centering)
            self.means = zeros((features.shape[1],))

        # Check if scaling should be performed
        if self.scale:

            # Compute the standard deviations of the feature columns
            self.deviations = std(features, axis=0)

        else:

            # Set all standard deviations to one (no scaling)
            self.deviations = ones((features.shape[1],))

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
        return array(
            [(features[:, i]-self.means[i])/self.deviations[i]
             for i in range(features.shape[1])]).T

    def compute_gradient(
            self,
            features):
        """
        Compute the gradient of the standard scaling transformation with \
        respect to the ``features``.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        Returns
        -------
        ndarray
            Values of the input feature gradients.
        """
        return array(
            [1/self.deviations[i] for i in range(features.shape[1])])
