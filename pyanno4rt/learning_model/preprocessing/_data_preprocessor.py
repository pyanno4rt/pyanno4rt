"""Data preprocessing pipeline."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import prod
from sklearn.pipeline import Pipeline

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model.preprocessing.transformers import (
    Equalizer, StandardScaler, Whitening)

# %% Class definition


class DataPreprocessor():
    """
    Data preprocessing pipeline class.

    Parameters
    ----------
    step_labels : tuple
        Tuple with the preprocessing pipeline elements (labels of the \
        respective preprocessing algorithm classes).

    Attributes
    ----------
    labels : tuple
        Tuple with the step labels.

    steps : tuple
        Tuple with the preprocessing algorithms.

    pipeline : object of class `Pipeline`
        Instance of the class `Pipeline`, which provides a preprocessing \
        pipeline (chain of transformation algorithms).
    """

    def __init__(
            self,
            step_labels):

        # Log a message about the preprocessor initialization
        Datahub().logger.display_info("Initializing data preprocessor ...")

        # Map the labels to the preprocessing algorithms
        catalogue = {'Equalizer': Equalizer(),
                     'StandardScaler': StandardScaler(),
                     'Whitening': Whitening()}

        # Get the preprocessing step labels
        self.labels = step_labels

        # Get the preprocessing algorithms related to the labels
        self.steps = tuple(catalogue[label] for label in self.labels)

        # Build the preprocessing pipeline
        self.pipeline = self.build()

    def build(self):
        """
        Build the preprocessing pipeline from the passed steps and step labels.

        Returns
        -------
        object of class `Pipeline`
            Instance of the class `Pipeline`, which provides a preprocessing \
            pipeline (chain of transformation algorithms).
        """
        # Log a message about the pipeline steps
        Datahub().logger.display_info("Building pipeline 'Input -> {} -> "
                                      "Output' ..."
                                      .format(' -> '.join(self.labels)))

        return Pipeline(list(zip(self.labels, self.steps)))

    def fit(
            self,
            features):
        """
        Fit the preprocessing pipeline with the input features.

        Parameters
        ----------
        features : ndarray
            Values of the input features.
        """
        self.pipeline.fit(features)

    def transform(
            self,
            features):
        """
        Transform the input features with the preprocessing pipeline.

        Returns
        -------
        ndarray
            Array of transformed feature values.
        """
        return self.pipeline.transform(features)

    def fit_transform(
            self,
            features):
        """
        Fit and transform the input features with the preprocessing pipeline.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        Returns
        -------
        ndarray
            Array of transformed feature values.
        """
        # Fit the preprocessing pipeline
        self.pipeline.fit(features)

        return self.pipeline.transform(features)

    def gradientize(
            self,
            features):
        """
        Compute the gradient of the preprocessing pipeline w.r.t the input \
        features.

        Parameters
        ----------
        features : ndarray
            Array of transformed feature values.

        Returns
        -------
        list
            Input feature gradients for the full preprocessing pipeline.
        """
        # Get the gradient for each preprocessing step
        step_gradients = tuple(
            step.compute_gradient(features) for step in self.steps
            if hasattr(step, 'compute_gradient') and callable(
                    getattr(step, 'compute_gradient')))

        return list(prod(x) for x in zip(*step_gradients))
