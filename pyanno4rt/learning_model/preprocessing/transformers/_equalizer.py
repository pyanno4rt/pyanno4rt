"""Equalization transformer."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import ones
from sklearn.base import BaseEstimator, TransformerMixin

# %% Class definition


class Equalizer(BaseEstimator, TransformerMixin):
    """
    Equalizer transformer class.

    This class provides methods to propagate the input features in an \
    unchanged matter, i.e., to build a "neutral" preprocessing pipeline.
    """

    # Set the algorithm label
    label = 'Equalizer'

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
        """
        return features

    def compute_gradient(
            self,
            features):
        """
        Compute the gradient of the equalization transformation with respect \
        to the ``features``.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        Returns
        -------
        ndarray
            Values of the input feature gradients.
        """
        return ones((features.shape[1],))
