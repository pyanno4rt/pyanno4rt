"""Isolation forest."""

# Author: Tim Ortkamp

# %% External package import

from sklearn.ensemble import IsolationForest as ScikitIsoForest

# %% Class definition


class IsolationForest():
    """
    Isolation forest class.

    This class provides methods to fit an isolation forest and transform the \
    input features and labels.

    Attributes
    ----------
    _classifier : str
        String indicating the preprocessing class.

    hyperparameters : dict
        Dictionary with the model hyperparameters.

    model : object of class :class:`~sklearn.ensemble.IsolationForest`
        The object used to represent the isolation forest.
    """

    def __init__(self):

        # Initialize the algorithm classifier
        self._classifier = 'outlier_removal'

        # Initialize the hyperparameters
        self.hyperparameters = {
            'n_estimators': 100,
            'max_samples': 'auto',
            'contamination': 'auto',
            'max_features': 1.0,
            'bootstrap': False,
            'n_jobs': None,
            'random_state': 14,
            'verbose': 0,
            'warm_start': False}

        # Initialize the isolation forest
        self.model = ScikitIsoForest(**self.hyperparameters)

    def fit(
            self,
            features,
            _):
        """
        Fit the isolation forest.

        Parameters
        ----------
        features : ndarray
            Feature values.
        """

        # Fit the isolation forest
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
