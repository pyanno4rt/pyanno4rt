"""Standard scaling transformer."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import array, mean, ones, std, zeros

# %% Class definition


class StandardScaler():
    """
    Standard scaling transformer class.

    This class provides methods to fit, transform and gradientize the input \
    features by their z-score.

    Parameters
    ----------
    center : bool, default=True
        Indicator for the centering of the data by the mean values.

    scale : bool, default=True
        Indicator for the scaling of the data by the standard deviations.

    Attributes
    ----------
    center : bool
        See 'Parameters'.

    scale : bool
        See 'Parameters'.

    means : ndarray
        Mean values of the features (if center is false, set to zeros).

    deviations : ndarray
        Standard deviations of the features (if scale is false, set to ones).
    """

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
            labels):
        """
        Fit the standard scaling transformer.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : None or ndarray
            Values of the input labels.
        """

        # Check if the features should be centered
        if self.center:

            # Compute the means of the features
            self.means = mean(features, axis=0)

        else:

            # Set the means to zero (no centering)
            self.means = zeros((features.shape[1],))

        # Check if the features should be scaled
        if self.scale:

            # Compute the standard deviations of the features
            self.deviations = std(features, axis=0)

        else:

            # Set the standard deviations to one (no scaling)
            self.deviations = ones((features.shape[1],))

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

        return array(
            [(features[:, index]-self.means[index])/self.deviations[index]
             for index in range(features.shape[1])]).T, labels

    def compute_gradient(
            self,
            features):
        """
        Compute the standard scaling transformer gradient w.r.t the input \
        features.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        Returns
        -------
        ndarray
            Value of the standard scaling transformer gradient.
        """

        return array(
            [1/self.deviations[index] for index in range(features.shape[1])])
