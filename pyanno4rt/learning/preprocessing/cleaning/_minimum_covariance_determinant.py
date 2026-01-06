"""Minimum covariance determinant."""

# Author: Tim Ortkamp

# %% External package import

from sklearn.covariance import EllipticEnvelope

# %% Class definition


class MinimumCovarianceDeterminant():
    """
    Minimum covariance determinant (MCD) class.

    This class provides methods to fit a MCD detector and transform the input \
    features and labels.

    Attributes
    ----------
    _classifier : str
        String indicating the preprocessing class.

    hyperparameters : dict
        Dictionary with the model hyperparameters.

    model : object of class :class:`~sklearn.covariance.EllipticEnvelope`
        The object used to represent the MCD detector.
    """

    def __init__(self):

        # Initialize the algorithm classifier
        self._classifier = 'outlier_removal'

        # Initialize the hyperparameters
        self.hyperparameters = {
            'store_precision': True,
            'assume_centered': False,
            'support_fraction': None,
            'contamination': 0.05,
            'random_state': 15}

        # Initialize the MCD detector
        self.model = EllipticEnvelope(**self.hyperparameters)

    def fit(
            self,
            features,
            _):
        """
        Fit the MCD detector.

        Parameters
        ----------
        features : ndarray
            Feature values.
        """

        # Fit the MCD detector
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

        # Get the inliers (= 1) and outliers (= -1)
        predictions = self.model.predict(features)

        # Get the filter mask
        mask = predictions == 1

        # Filter the features
        features = features[mask]

        # Check if label values are provided
        if labels is not None:

            # Filter the labels
            labels = labels[mask]

        return features, labels

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
