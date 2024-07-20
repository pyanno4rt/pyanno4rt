"""Decision tree model."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pickle import dump, load

from hyperopt import hp
from sklearn.tree import DecisionTreeClassifier

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model.frequentist import MachineLearningModel

# %% Class definition


class DecisionTreeModel(MachineLearningModel):
    """
    Decision tree model class.

    This class enables building an individual preprocessing pipeline, \
    fitting, making predictions, inspecting, and evaluating the predictive \
    performance of a decision tree model.

    See the machine learning model template class \
        :class:`~pyanno4rt.learning_model.frequentist._machine_learning_model.MachineLearningModel`
    for information on the parameters and attributes.

    .. note:: Currently, the hyperparameter search space for the decision \
        tree model includes:

            - 'criterion' : measure for the quality of a split
            - 'splitter' : splitting strategy at each node
            - 'max_depth' : maximum depth of the tree
            - 'min_samples_split' : minimum number of samples required to \
                split an internal node
            - 'min_samples_leaf' : minimum number of samples required at a \
                leaf node
            - 'min_weight_fraction_leaf' : minimum weighted fraction of the \
                sum of weights required at each node
            - 'max_features' : number of features considered at each split
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
            'criterion': tune_space.get('criterion', ['gini', 'entropy']),
            'splitter': tune_space.get('splitter', ['best', 'random']),
            'max_depth': tune_space.get('max_depth', list(range(1, 21))),
            'min_samples_split': tune_space.get(
                'min_samples_split', [0.0, 1.0]),
            'min_samples_leaf': tune_space.get('min_samples_leaf', [0.0, 0.5]),
            'min_weight_fraction_leaf': tune_space.get(
                'min_weight_fraction_leaf', [0.0, 0.5]),
            'max_features': tune_space.get('max_features', list(range(
                    1, dataset['feature_values'].shape[1]+1))),
            'class_weight': tune_space.get('class_weight', [None, 'balanced']),
            'ccp_alpha': tune_space.get('ccp_alpha', [0.0, 1.0])}

        # Configure the hyperopt search space
        hp_space = {
            'criterion': hp.choice('criterion', tune_space['criterion']),
            'splitter': hp.choice('splitter', tune_space['splitter']),
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
            'random_state': 42,
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'monotonic_cst': None}

        return hyperparameters

    def get_model_fit(
            self,
            features,
            labels,
            hyperparameters):
        """
        Get the decision tree model fit.

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
            :class:`~sklearn.tree.DecisionTreeClassifier`
            The fitted object used to represent the prediction model.
        """

        # Initialize the decision tree model
        prediction_model = DecisionTreeClassifier(**hyperparameters)

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
            Array of input feature values.

        predictor : object of class \
            :class:`~sklearn.tree.DecisionTreeClassifier`
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
        Read the decision tree model from the model file path.

        Returns
        -------
        object of class :class:`~sklearn.tree.DecisionTreeClassifier`
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
        Write the decision tree model to the model file path.

        Parameters
        ----------
        prediction_model : object of class \
            :class:`~sklearn.tree.DecisionTreeClassifier`
            The object used to represent the prediction model.
        """

        # Open a file stream
        with open(self.model_path, 'wb') as file:

            # Dump the model to the model file path
            dump(prediction_model, file)
