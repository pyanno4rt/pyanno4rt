"""Logistic regression model."""

# Author: Tim Ortkamp

# %% External package import

from pickle import dump, load

from hyperopt import hp
from sklearn.linear_model import LogisticRegression

# %% Internal package import

from pyanno4rt.learning.models import MachineLearningModel
from pyanno4rt.logging import get_logger

# %% Class definition


class LogisticRegressionModel(MachineLearningModel):
    """
    Logistic regression model class.

    This class enables building an individual preprocessing pipeline, \
    fitting, making predictions, inspecting, and evaluating the predictive \
    performance of a logistic regression model.

    See the machine learning model template class \
        :class:`~pyanno4rt.learning._machine_learning_model.MachineLearningModel`
    for information on the parameters and attributes.
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

        # Get the internal hyperparameter search space
        tune_space_dict = tune_space.to_dict()

        # Configure the hyperopt search space
        hp_space = {
            'regularization': hp.choice(
                'regularization', [
                    {'penalty': None,
                     'solver': hp.choice(
                         'solver_None',
                         ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag'])},
                    *[{'penalty': norm,
                       'solver': hp.choice(
                           f'solver_{norm}',
                           ['liblinear', 'saga'] if norm == 'l1'
                           else [
                               'lbfgs', 'liblinear', 'newton-cg',
                               'newton-cholesky', 'sag', 'saga']),
                       'C': hp.uniform(
                           f'C_{norm}', tune_space_dict['C'][0],
                           tune_space_dict['C'][1])
                       }
                      if norm != 'elasticnet' else
                      {'penalty': 'elasticnet',
                       'l1_ratio': 0.5,
                       'solver': hp.choice(
                           f'solver_{norm}', ['saga']),
                       'C': hp.uniform(
                           f'C_{norm}', tune_space_dict['C'][0],
                           tune_space_dict['C'][1])
                       }
                      for norm in tune_space_dict['penalty']]
                    ]),
            'tol': hp.choice('tol', tune_space_dict['tol']),
            'class_weight': hp.choice(
                'class_weight', tune_space_dict['class_weight'])}

        # Initialize the superclass
        super().__init__(
            model_label, model_folder_path, dataset, preprocessing_steps,
            tune_space_dict, hp_space, tune_evaluations, tune_score,
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
            regularization = {key: proposal.get(key) for key in (
                'penalty', 'solver', 'l1_ratio', 'C')}

        # Build the hyperparameter dictionary
        hyperparameters = {
            **regularization,
            'dual': False,
            'tol': proposal['tol'],
            'fit_intercept': True,
            'intercept_scaling': 1,
            'class_weight': proposal['class_weight'],
            'random_state': 42,
            'max_iter': 10**6,
            'verbose': 0,
            'warm_start': False,
            'n_jobs': -1 if regularization['solver'] != 'liblinear' else 1}

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
            The object used to represent the pre-fitted prediction model.
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

    def import_model(self):
        """
        Import the logistic regression model.

        Returns
        -------
        object of class :class:`~sklearn.linear_model.LogisticRegression`
            The object used to represent the prediction model.
        """

        # Log a message about the model file reading
        get_logger().info("Reading '%s' model from file ...", self.model_label)

        return load(open(self.model_path, 'rb'))

    def export_model(self):
        """Export the logistic regression model."""

        # Open a file stream
        with open(self.model_path, 'wb') as file:

            # Dump the model
            dump(self.prediction_model, file)
