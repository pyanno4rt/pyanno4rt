"""Logistic regression model."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pickle import dump, load

from hyperopt import hp
from sklearn.linear_model import LogisticRegression

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model.frequentist import MachineLearningModel

# %% Class definition


class LogisticRegressionModel(MachineLearningModel):
    """
    Logistic regression model class.

    This class enables building an individual preprocessing pipeline, \
    fitting, making predictions, inspecting, and evaluating the predictive \
    performance of a logistic regression model.

    See the machine learning model template class \
        :class:`~pyanno4rt.learning_model.frequentist._machine_learning_model.MachineLearningModel`
    for information on the parameters and attributes.

    .. note:: Currently, the hyperparameter search space for the logistic \
        regression model includes:

            - 'C' : inverse of the regularization strength
            - 'penalty' : norm of the penalty function
            - 'tol' : tolerance for stopping criteria
            - 'class_weight' : weights associated with the classes
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
            'penalty': tune_space.get('penalty', ['l1', 'l2', 'elasticnet']),
            'tol': tune_space.get('tol', [1e-4, 1e-5, 1e-6]),
            'C': tune_space.get('C', [2**-5, 2**10]),
            'class_weight': tune_space.get('class_weight', [None, 'balanced'])}

        # Configure the hyperopt search space
        hp_space = {
            'regularization': hp.choice(
                'regularization', [
                    {'penalty': None, 'C': 0.01},
                    *[{'penalty': norm, 'C': hp.uniform(
                        f'C_{norm}', tune_space['C'][0], tune_space['C'][1])}
                        for norm in tune_space['penalty']]]),
            'tol': hp.choice('tol', tune_space['tol']),
            'class_weight': hp.choice(
                'class_weight', tune_space['class_weight'])}

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

        # Check if the proposal has a regularization subdictionary
        if 'regularization' in proposal:

            # Get the unpacked regularization parameters
            regularization = {**proposal['regularization']}

        else:

            # Get the regularization parameters directly
            regularization = {key: proposal[key] for key in ('C', 'penalty')}

        # Build the hyperparameter dictionary
        hyperparameters = {
            **regularization,
            'dual': False,
            'tol': proposal['tol'],
            'fit_intercept': True,
            'intercept_scaling': 1,
            'class_weight': proposal['class_weight'],
            'random_state': 42,
            'solver': 'saga',
            'max_iter': 10**6,
            'verbose': 0,
            'warm_start': False,
            'n_jobs': -1,
            'l1_ratio': 0.5}

        return hyperparameters

    def get_model_fit(
            self,
            features,
            labels,
            hyperparameters):
        """
        Get the logistic regression model fit.

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
            :class:`~sklearn.linear_model.LogisticRegression`
            The fitted object used to represent the prediction model.
        """

        # Initialize the logistic regression model
        prediction_model = LogisticRegression(**hyperparameters)

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
            :class:`~sklearn.linear_model.LogisticRegression`
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
        Read the logistic regression model from the model file path.

        Returns
        -------
        object of class :class:`~sklearn.linear_model.LogisticRegression`
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
        Write the logistic regression model to the model file path.

        Parameters
        ----------
        prediction_model : object of class \
            :class:`~sklearn.linear_model.LogisticRegression`
            The object used to represent the prediction model.
        """

        # Open a file stream
        with open(self.model_path, 'wb') as file:

            # Dump the model to the model file path
            dump(prediction_model, file)
