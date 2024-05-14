"""Identity transformer."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import ones

# %% Class definition


class Identity():
    """
    Identity transformer class.

    This class provides methods to fit, transform and gradientize the input \
    features by their identity (default preprocessing step).
    """

    def fit(
            self,
            features,
            labels):
        """
        Fit the identity transformer.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : None or ndarray
            Values of the input labels.
        """

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

        return features, labels

    def compute_gradient(
            self,
            features):
        """
        Compute the identity transformer gradient w.r.t the input features.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        Returns
        -------
        ndarray
            Value of the identity transformer gradient.
        """

        return ones(features.shape)
