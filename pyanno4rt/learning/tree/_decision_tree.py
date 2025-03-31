"""Decision tree model."""

# Author: Tim Ortkamp

# %% External package import

from pickle import dump, load

from hyperopt import hp
from sklearn.tree import DecisionTreeClassifier

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning import MachineLearningModel
from pyanno4rt.learning.tree import OptimizableDecisionTree

# %% Class definition


class DecisionTreeModel(MachineLearningModel):
    """
    Decision tree model class.

    This class enables building an individual preprocessing pipeline, \
    fitting, making predictions, inspecting, and evaluating the predictive \
    performance of a decision tree model.

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

        # Check if the maximum number of features is set to the default
        if tune_space_dict['max_features'] == [0]:

            # Adjust the maximum number of features by the dataset
            tune_space_dict['max_features'] = [len(dataset['feature_names'])]

        # Configure the hyperopt search space
        hp_space = {
            'criterion': hp.choice('criterion', tune_space_dict['criterion']),
            'splitter': hp.choice('splitter', tune_space_dict['splitter']),
            'max_depth': hp.choice('max_depth', tune_space_dict['max_depth']),
            'min_samples_split': hp.uniform(
                'min_samples_split', tune_space_dict['min_samples_split'][0],
                tune_space_dict['min_samples_split'][1]),
            'min_samples_leaf': hp.uniform(
                'min_samples_leaf', tune_space_dict['min_samples_leaf'][0],
                tune_space_dict['min_samples_leaf'][1]),
            'min_weight_fraction_leaf': hp.uniform(
                'min_weight_fraction_leaf',
                tune_space_dict['min_weight_fraction_leaf'][0],
                tune_space_dict['min_weight_fraction_leaf'][1]),
            'max_features': hp.choice(
                'max_features', tune_space_dict['max_features']),
            'class_weight': hp.choice(
                'class_weight', tune_space_dict['class_weight']),
            'ccp_alpha': hp.uniform(
                'ccp_alpha', tune_space_dict['ccp_alpha'][0],
                tune_space_dict['ccp_alpha'][1])}

        # Initialize the superclass
        super().__init__(
            model_label, model_folder_path, dataset, preprocessing_steps,
            tune_space_dict, hp_space, tune_evaluations, tune_score,
            inspect_model, evaluate_model, display_options)

        # Get the optimization surrogate of the decision tree model
        self.optimization_model = self.get_optimization_model()

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
            The object used to represent the pre-fitted prediction model.
        """

        # Initialize the decision tree model
        prediction_model = DecisionTreeClassifier(**hyperparameters)

        # Fit the model with the training data
        prediction_model.fit(features, labels)

        return prediction_model

    def get_optimization_model(self):
        """
        Get the decision tree optimization model.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.tree._optimizable_decision_tree.OptimizableDecisionTree`
            The object used to represent the optimization model.
        """

        # Initialize the optimizable decision tree
        optimization_model = OptimizableDecisionTree()

        # Read the path information from the pre-fitted decision tree
        optimization_model.traverse(self.prediction_model)

        return optimization_model

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

    def import_model(self):
        """
        Import the decision tree model.

        Returns
        -------
        object of class :class:`~sklearn.tree.DecisionTreeClassifier`
            The object used to represent the prediction model.
        """

        # Log a message about the model file reading
        Datahub().logger.display_info(
            f'Reading "{self.model_label}" model from file ...')

        return load(open(self.model_path, 'rb'))

    def export_model(self):
        """
        Export the decision tree model."""

        # Open a file stream
        with open(self.model_path, 'wb') as file:

            # Dump the model
            dump(self.prediction_model, file)
