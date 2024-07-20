"""Random forest model."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pickle import dump, load

from hyperopt import hp
from sklearn.ensemble import RandomForestClassifier

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model.frequentist import MachineLearningModel

# %% Class definition


class RandomForestModel(MachineLearningModel):
    """
    Random forest model class.

    This class enables building an individual preprocessing pipeline, \
    fitting, making predictions, inspecting, and evaluating the predictive \
    performance of a random forest model.

    See the machine learning model template class \
        :class:`~pyanno4rt.learning_model.frequentist._machine_learning_model.MachineLearningModel`
    for information on the parameters and attributes.

    .. note:: Currently, the hyperparameter search space for the random \
        model includes:

            - 'n_estimators' : number of trees in the forest
            - 'criterion' : measure for the quality of a split
            - 'max_depth' : maximum depth of the tree
            - 'min_samples_split' : minimum number of samples required to \
                split an internal node
            - 'min_samples_leaf' : minimum number of samples required at a \
                leaf node
            - 'min_weight_fraction_leaf' : minimum weighted fraction of the \
                sum of weights required at each node
            - 'max_features' : number of features considered at each split
            - 'bootstrap' : indicator for the use of bootstrap samples to \
                build the trees
            - 'class_weight' : weights associated with the classes
            - 'ccp_alpha' : complexity parameter for minimal cost-complexity \
                pruning
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
            'n_estimators': tune_space.get('n_estimators', list(range(1, 51))),
            'criterion': tune_space.get('criterion', ['gini', 'entropy']),
            'max_depth': tune_space.get('max_depth', list(range(1, 21))),
            'min_samples_split': tune_space.get(
                'min_samples_split', [0.0, 1.0]),
            'min_samples_leaf': tune_space.get('min_samples_leaf', [0.0, 0.5]),
            'min_weight_fraction_leaf': tune_space.get(
                'min_weight_fraction_leaf', [0.0, 0.5]),
            'max_features': tune_space.get('max_features', list(range(
                    1, dataset['feature_values'].shape[1]+1))),
            'bootstrap': tune_space.get('bootstrap', [False, True]),
            'class_weight': tune_space.get('class_weight', [None, 'balanced']),
            'ccp_alpha': tune_space.get('ccp_alpha', [0.0, 1.0])}

        # Configure the hyperopt search space
        hp_space = {
            'n_estimators': hp.choice(
                'n_estimators', tune_space['n_estimators']),
            'criterion': hp.choice('criterion', tune_space['criterion']),
            'max_depth': hp.choice('max_depth', tune_space['max_depth']),
            'min_samples_split': hp.uniform(
                'min_samples_split', tune_space['min_samples_split'][0],
                tune_space['min_samples_split'][1]),
            'min_samples_leaf': hp.uniform(
                'min_samples_leaf', tune_space['min_samples_leaf'][0],
                tune_space['min_samples_leaf'][1]),
            'min_weight_fraction_leaf': hp.uniform(
                'min_weight_fraction_leaf',
                tune_space['min_weight_fraction_leaf'][0],
                tune_space['min_weight_fraction_leaf'][1]),
            'max_features': hp.choice(
                'max_features', tune_space['max_features']),
            'bootstrap': hp.choice('bootstrap', tune_space['bootstrap']),
            'class_weight': hp.choice(
                'class_weight', tune_space['class_weight']),
            'ccp_alpha': hp.uniform(
                'ccp_alpha', tune_space['ccp_alpha'][0],
                tune_space['ccp_alpha'][1])}

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
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'oob_score': False,
            'n_jobs': -1,
            'random_state': 42,
            'verbose': 0,
            'warm_start': False,
            'max_samples': None,
            'monotonic_cst': None}

        return hyperparameters

    def get_model_fit(
            self,
            features,
            labels,
            hyperparameters):
        """
        Get the random forest model fit.

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
            :class:`~sklearn.ensemble.RandomForestClassifier`
            The fitted object used to represent the prediction model.
        """

        # Initialize the random forest model
        prediction_model = RandomForestClassifier(**hyperparameters)

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
            :class:`~sklearn.ensemble.RandomForestClassifier`
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
        Read the random forest model from the model file path.

        Returns
        -------
        object of class :class:`~sklearn.ensemble.RandomForestClassifier`
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
        Write the random forest model to the model file path.

        Parameters
        ----------
        prediction_model : object of class \
            :class:`~sklearn.ensemble.RandomForestClassifier`
            The object used to represent the prediction model.
        """

        # Open a file stream
        with open(self.model_path, 'wb') as file:

            # Dump the model to the model file path
            dump(prediction_model, file)
