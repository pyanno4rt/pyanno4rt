"""Support vector machine model."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pickle import dump, load

from hyperopt import hp
from sklearn.svm import SVC

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model.frequentist import MachineLearningModel

# %% Class definition


class SupportVectorMachineModel(MachineLearningModel):
    """
    Support vector machine model class.

    This class enables building an individual preprocessing pipeline, \
    fitting, making predictions, inspecting, and evaluating the predictive \
    performance of a support vector machine model.

    See the machine learning model template class \
        :class:`~pyanno4rt.learning_model.frequentist._machine_learning_model.MachineLearningModel`
    for information on the parameters and attributes.

    .. note:: Currently, the hyperparameter search space for the support \
        vector machine model includes:

            - 'C' : inverse proportional of the regularization strength
            - 'kernel' : kernel type
            - 'degree' : degree of the polynomial kernel function
            - 'gamma' : kernel coefficient for 'rbf', 'poly' and 'sigmoid'
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
            'C': tune_space.get('C', [2**-5, 2**10]),
            'kernel': tune_space.get(
                'kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'degree': tune_space.get('degree', [3, 4, 5, 6]),
            'gamma': tune_space.get('gamma', [2**-15, 2**3]),
            'tol': tune_space.get('tol', [1e-4, 1e-5, 1e-6]),
            'class_weight': tune_space.get('class_weight', [None, 'balanced'])}

        # Configure the hyperopt search space
        hp_space = {
            'C': hp.uniform('C', tune_space['C'][0], tune_space['C'][1]),
            'kernel': hp.choice('kernel', tune_space['kernel']),
            'degree': hp.choice('degree', tune_space['degree']),
            'gamma': hp.uniform(
                'gamma', tune_space['gamma'][0], tune_space['gamma'][1]),
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

        # Build the hyperparameter dictionary
        hyperparameters = {
            'C': proposal['C'],
            'kernel': proposal['kernel'],
            'degree': proposal['degree'],
            'gamma': proposal['gamma'],
            'coef0': 0.0,
            'shrinking': True,
            'probability': True,
            'tol': proposal['tol'],
            'cache_size': 200,
            'class_weight': proposal['class_weight'],
            'verbose': False,
            'max_iter': -1,
            'decision_function_shape': 'ovr',
            'break_ties': False,
            'random_state': 42}

        return hyperparameters

    def get_model_fit(
            self,
            features,
            labels,
            hyperparameters):
        """
        Get the support vector machine model fit.

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
        prediction_model : object of class :class:`~sklearn.svm.SVC`
            The fitted object used to represent the prediction model.
        """

        # Initialize the support vector machine model
        prediction_model = SVC(**hyperparameters)

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

        predictor : object of class :class:`~sklearn.svm.SVC`
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
        Read the support vector machine model from the model file path.

        Returns
        -------
        object of class :class:`~sklearn.svm.SVC`
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
        Write the support vector machine model to the model file path.

        Parameters
        ----------
        prediction_model : object of class :class:`~sklearn.svm.SVC`
            The object used to represent the prediction model.
        """

        # Open a file stream
        with open(self.model_path, 'wb') as file:

            # Dump the model to the model file path
            dump(prediction_model, file)
