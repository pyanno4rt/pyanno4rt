"""Standard scaling."""

# Author: Tim Ortkamp

# %% External package import

from numpy import array, diag, mean, ones, std, zeros

# %% Class definition


class StandardScaler():
    """
    Standard scaling class.

    This class provides methods to fit a standard scaler, transform and \
    gradientize the input features.

    Parameters
    ----------
    center : bool, default=True
        Indicator for the centering of the data by the mean values.

    scale : bool, default=True
        Indicator for the scaling of the data by the standard deviations.

    Attributes
    ----------
    _classifier : str
        String indicating the preprocessing class.

    center : bool
        See 'Parameters'.

    scale : bool
        See 'Parameters'.

    means : ndarray
        Mean values of the features (if center is false, set to zeros).

    deviations : ndarray
        Standard deviations of the features (if scale is false, set to ones).
    """

    # Initialize the algorithm classifier
    _classifier = 'scaling'

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
            _):
        """
        Fit the standard scaler.

        Parameters
        ----------
        features : ndarray
            Feature values.
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
            Input feature values.

        Returns
        -------
        ndarray
            Input gradient.
        """

        return diag(array(
            [1/self.deviations[index] for index in range(features.shape[1])]))
