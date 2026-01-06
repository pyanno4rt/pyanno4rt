"""Principal component analysis."""

# Author: Tim Ortkamp

# %% External package import

from sklearn.decomposition import PCA as ScikitPCA

# %% Class definition


class PrincipalComponentAnalysis():
    """
    Principal component analysis (PCA) class.

    This class provides methods to fit a princial component analyzer, \
    transform and gradientize the input features.

    Attributes
    ----------
    _classifier : str
        String indicating the preprocessing class.

    hyperparameters : dict
        Dictionary with the model hyperparameters.

    model : object of class :class:`~sklearn.decomposition.PCA`
        The object used to represent the principal component analyzer.
    """

    def __init__(self):

        # Initialize the algorithm classifier
        self._classifier = 'dimensionality_reduction'

        # Initialize the hyperparameters
        self.hyperparameters = {
            'n_components': 0.99,
            'copy': True,
            'whiten': False,
            'svd_solver': 'auto',
            'tol': 0.0,
            'iterated_power': 'auto',
            'n_oversamples': 10,
            'power_iteration_normalizer': 'auto',
            'random_state': None}

        # Initialize the PCA model
        self.model = ScikitPCA(**self.hyperparameters)

    def fit(
            self,
            features,
            _):
        """
        Fit the PCA model.

        Parameters
        ----------
        features : ndarray
            Feature values.
        """

        # Fit the PCA model
        self.model.fit(features)

        return self

    def transform(
            self,
            features,
            labels=None):
        """
        Transform the input features/labels.

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

        return self.model.transform(features), labels

    def fit_transform(
            self,
            features,
            labels=None):
        """
        Fit and transform the input features/labels.

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

        return self.model.fit_transform(features, labels)

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

        return self.model.components_.T
