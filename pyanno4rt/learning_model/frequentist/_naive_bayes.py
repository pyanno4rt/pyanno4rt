"""Naive Bayes model."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pickle import dump, load

from hyperopt import hp
from numpy import mean
from sklearn.naive_bayes import GaussianNB

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model.frequentist import MachineLearningModel

# %% Class definition


class NaiveBayesModel(MachineLearningModel):
    """
    Naive Bayes model class.

    This class enables building an individual preprocessing pipeline, \
    fitting, making predictions, inspecting, and evaluating the predictive \
    performance of a naive Bayes model.

    See the machine learning model template class \
        :class:`~pyanno4rt.learning_model.frequentist._machine_learning_model.MachineLearningModel`
    for information on the parameters and attributes.

    .. note:: Currently, the hyperparameter search space for the naive Bayes \
        model includes:

            - 'priors' : prior probabilities of the classes
            - 'var_smoothing' : portion of the largest variance of all \
                features added to variances for calculation stability
    """

    def __init__(
            self,
            model_label,
            model_folder_path,
            dataset,
            preprocessing_steps,
            tune_space,
            tune_evaluations,
            tune_score,
            inspect_model,
            evaluate_model,
            display_options):

        # Configure the internal hyperparameter search space
        tune_space = {
            'priors': tune_space.get(
                'priors', [None, [1-mean(dataset['label_values']),
                                  mean(dataset['label_values'])]]),
            'var_smoothing': tune_space.get('var_smoothing', [1e-12, 1])}

        # Configure the hyperopt search space
        hp_space = {
            'priors': hp.choice('priors', tune_space['priors']),
            'var_smoothing': hp.uniform(
                'var_smoothing', tune_space['var_smoothing'][0],
                tune_space['var_smoothing'][1])}

        # Initialize the superclass
        super().__init__(
            model_label, model_folder_path, dataset, preprocessing_steps,
            tune_space, hp_space, tune_evaluations, tune_score,
            inspect_model, evaluate_model, display_options)

    def get_hyperparameter_set(
            self,
            proposal):
        """
        Get the hyperparameter set.

        Parameters
        ----------
        proposal : dict
            Proposal for the hyperparameter set.

        Returns
        -------
        hyperparameters : dict
            Dictionary with the values of the hyperparameters.
        """

        # Build the hyperparameter dictionary
        hyperparameters = {**proposal}

        return hyperparameters

    def get_model_fit(
            self,
            features,
            labels,
            hyperparameters):
        """
        Get the naive Bayes model fit.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : ndarray
            Values of the input labels.

        hyperparameters : dict
            Dictionary with the values of the hyperparameters.

        Returns
        -------
        prediction_model : object of class \
            :class:`~sklearn.naive_bayes.GaussianNB`
            The fitted object used to represent the prediction model.
        """

        # Initialize the naive Bayes model
        prediction_model = GaussianNB(**hyperparameters)

        # Fit the model with the training data
        prediction_model.fit(features, labels)

        return prediction_model

    def predict(
            self,
            features,
            predictor):
        """
        Predict the label values.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        predictor : object of class \
            :class:`~sklearn.naive_bayes.GaussianNB`
            The object used to represent the prediction model.

        Returns
        -------
        float or ndarray
            Value(s) of the predicted label(s).
        """

        # Check if the feature array has only a single row
        if features.shape[0] == 1:

            # Return a single label prediction value
            return predictor.predict_proba(features)[0][1]

        # Else, return an array with label predictions
        return predictor.predict_proba(features)[:, 1]

    def read_model_from_file(self):
        """
        Read the naive Bayes model from the model file path.

        Returns
        -------
        object of class :class:`~sklearn.naive_bayes.GaussianNB`
            The object used to represent the prediction model.
        """

        # Log a message about the model file reading
        Datahub().logger.display_info(
            f'Reading "{self.model_label}" model from file ...')

        return load(open(self.model_path, 'rb'))

    def write_model_to_file(
            self,
            prediction_model):
        """
        Write the naive Bayes model to the model file path.

        Parameters
        ----------
        prediction_model : object of class \
            :class:`~sklearn.naive_bayes.GaussianNB`
            The object used to represent the prediction model.
        """

        # Open a file stream
        with open(self.model_path, 'wb') as file:

            # Dump the model to the model file path
            dump(prediction_model, file)
