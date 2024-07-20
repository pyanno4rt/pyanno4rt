"""K-nearest neighbors model."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pickle import dump, load

from hyperopt import hp
from sklearn.neighbors import KNeighborsClassifier

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model.frequentist import MachineLearningModel

# %% Class definition


class KNeighborsModel(MachineLearningModel):
    """
    K-nearest neighbors model class.

    This class enables building an individual preprocessing pipeline, \
    fitting, making predictions, inspecting, and evaluating the predictive \
    performance of a k-nearest neighbors model.

    See the machine learning model template class \
        :class:`~pyanno4rt.learning_model.frequentist._machine_learning_model.MachineLearningModel`
    for information on the parameters and attributes.

    .. note:: Currently, the hyperparameter search space for the k-nearest \
        neighbors model includes:

            - 'n_neighbors' : number of neighbors
            - 'weights' : weight function for prediction
            - 'leaf_size' : leaf size for BallTree or KDTree
            - 'p' : power parameter for the Minkowski metric
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
            'n_neighbors': tune_space.get('n_neighbors', list(range(
                    1, round(0.5*dataset['feature_values'].shape[0])))),
            'weights': tune_space.get('weights', ['uniform', 'distance']),
            'leaf_size': tune_space.get('leaf_size', list(range(1, 501))),
            'p': tune_space.get('p', [1, 2, 3])}

        # Configure the hyperopt search space
        hp_space = {
            'n_neighbors': hp.choice('n_neighbors', tune_space['n_neighbors']),
            'weights': hp.choice('weights', tune_space['weights']),
            'leaf_size': hp.choice('leaf_size', tune_space['leaf_size']),
            'p': hp.choice('p', tune_space['p'])}

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
            **proposal,
            'algorithm': 'auto',
            'metric': 'minkowski',
            'metric_params': None,
            'n_jobs': -1}

        return hyperparameters

    def get_model_fit(
            self,
            features,
            labels,
            hyperparameters):
        """
        Get the k-nearest neighbors model fit.

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
            :class:`~sklearn.neighbors.KNeighborsClassifier`
            The fitted object used to represent the prediction model.
        """

        # Initialize the k-nearest neighbors model
        prediction_model = KNeighborsClassifier(**hyperparameters)

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
            :class:`~sklearn.neighbors.KNeighborsClassifier`
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
        Read the k-nearest neighbors model from the model file path.

        Returns
        -------
        object of class :class:`~sklearn.neighbors.KNeighborsClassifier`
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
        Write the k-nearest neighbors model to the model file path.

        Parameters
        ----------
        prediction_model : object of class \
            :class:`~sklearn.neighbors.KNeighborsClassifier`
            The object used to represent the prediction model.
        """

        # Open a file stream
        with open(self.model_path, 'wb') as file:

            # Dump the model to the model file path
            dump(prediction_model, file)
