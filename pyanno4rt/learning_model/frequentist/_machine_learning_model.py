"""Machine learning model template."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from json import dump as jdump, load as jload
from os.path import exists

from abc import ABCMeta, abstractmethod
from functools import partial
from hyperopt import fmin, space_eval, STATUS_FAIL, STATUS_OK, Trials, tpe
from numpy import array, empty, where
from sklearn.metrics import roc_auc_score

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model.evaluation import ModelEvaluator
from pyanno4rt.learning_model.inspection import ModelInspector
from pyanno4rt.learning_model.losses import loss_map as score_loss_map
from pyanno4rt.learning_model.preprocessing import DataPreprocessor
from pyanno4rt.tools import compare_dictionaries

# %% Class definition


class MachineLearningModel(metaclass=ABCMeta):
    """
    Machine learning model template class.

    Parameters
    ----------
    model_label : str
        Label for the model.

    model_folder_path : None or str
        Path to a folder for loading an external model.

    dataset : dict
        Dictionary with the base data information.

    preprocessing_steps : list
        Sequence of labels associated with preprocessing algorithms to \
        preprocess the input features.

    tune_space : dict
        Internal search space for the Bayesian hyperparameter optimization.

    hp_space : dict
        Hyperopt search space for the Bayesian hyperparameter optimization.

    tune_evaluations : int
        Number of evaluation steps (trials) for the Bayesian \
        hyperparameter optimization.

    tune_score : {'AUC', 'Brier score', 'Logloss'}
        Scoring function for the evaluation of the hyperparameter set \
        candidates.

    inspect_model : bool
        Indicator for the inspection of the model.

    evaluate_model : bool
        Indicator for the evaluation of the model.

    display_options : dict
        Dictionary with the graph and KPI display options.

    architecture : None or str, default=None
        Type of architecture (only used in neural networks).

    max_hidden_layers : None or int, default=None
        Maximum number of hidden layers (only used in neural networks).

    Attributes
    ----------
    model_label : str
        See 'Parameters'.

    model_folder_path : None or str
        See 'Parameters'.

    preprocessing_steps : list
        See 'Parameters'.

    model_path : None or str
        Path for storing and retrieving the model.

    configuration_path : None or str
        Path for storing and retrieving the configuration dictionary.

    hyperparameter_path : None or str
        Path for storing and retrieving the hyperparameter dictionary.

    configuration : dict
        Dictionary with information on the model configuration.

    hp_space : dict
        See 'Parameters'.

    step : int
        Counter variable for the tuning evaluations.

    preprocessor : object of class \
        :class:`~pyanno4rt.learning_model.preprocessing._data_preprocessor.DataPreprocessor`
        The object used to build the preprocessing pipeline, transform the \
        data, and return the input gradients of the preprocessing algorithms.

    preprocessed_features : ndarray
        Values of the preprocessed input features.

    preprocessed_labels : ndarray
        Values of the preprocessed input labels.

    updated_model : bool
        Indicator for the update status of the model.

    prediction_model : object
        The object used to represent the prediction model.

    hyperparameters : dict
        Dictionary with the values of the hyperparameters.

    inspector : object of class \
        :class:`~pyanno4rt.learning_model.inspection._model_inspector.ModelInspector`
        The object used to inspect the model.

    evaluator : object of class \
        :class:`~pyanno4rt.learning_model.evaluation._model_evaluator.ModelEvaluator`
        The object used to evaluate the model.
    """

    def __init__(
            self,
            model_label,
            model_folder_path,
            dataset,
            preprocessing_steps,
            tune_space,
            hp_space,
            tune_evaluations,
            tune_score,
            inspect_model,
            evaluate_model,
            display_options,
            architecture=None,
            max_hidden_layers=None):

        # Initialize the datahub
        hub = Datahub()

        # Get the instance attributes from the arguments
        self.model_label, self.model_folder_path, self.preprocessing_steps = (
            model_label, model_folder_path, preprocessing_steps)

        # Initialize the file paths
        self.model_path, self.configuration_path, self.hyperparameter_path = (
            None, None, None)

        # Build the configuration dictionary with the modeling information
        self.configuration = {
            'feature_names': dataset['feature_names'],
            'feature_values': dataset['feature_values'],
            'label_name': dataset['label_name'],
            'label_values': dataset['label_values'],
            'time_variable_name': dataset['time_variable_name'],
            'time_variable_values': dataset['time_variable_values'],
            'tune_folds': dataset['tune_folds'],
            'oof_folds': dataset['oof_folds'],
            'label_bounds': dataset['label_bounds'],
            'label_viewpoint': dataset.get('label_viewpoint'),
            'preprocessing_steps': preprocessing_steps,
            'architecture': architecture,
            'max_hidden_layers': max_hidden_layers,
            'tune_space': tune_space,
            'tune_evaluations': tune_evaluations,
            'tune_score': tune_score}

        # Get the hyperopt search space
        self.hp_space = hp_space

        # Initialize the step counter for the hyperparameter search
        self.step = None

        # Initialize the data preprocessor
        self.preprocessor = DataPreprocessor(preprocessing_steps)

        # Initialize the boolean flag to indicate model updates
        self.updated_model = False

        # Fit the data preprocessor and transform the input feature values
        self.preprocessed_features, self.preprocessed_labels = (
            self.preprocessor.fit_transform(
                dataset['feature_values'], dataset['label_values']))

        # Get the logistic regression model and its hyperparameters
        self.prediction_model, self.hyperparameters = self.get_model(
            dataset['feature_values'], dataset['label_values'])

        # Check if the model has been updated or not yet registered
        if self.updated_model or self.model_label not in hub.model_instances:

            # Add the model instance to the datahub
            hub.model_instances[self.model_label] = {
                'prediction_model': self.prediction_model,
                'configuration': self.configuration,
                'hyperparameters': self.hyperparameters}

        # Check if the model should be inspected
        if inspect_model:

            # Initialize the model inspector
            self.inspector = ModelInspector(model_label)

            # Inspect the model
            self.inspect(dataset['feature_values'], dataset['label_values'],
                         dataset['oof_folds'])

        # Check if the model should be evaluated
        if evaluate_model:

            # Initialize the model evaluator
            self.evaluator = ModelEvaluator(model_label)

            # Evaluate the model
            self.evaluate(dataset['feature_values'], dataset['label_values'])

        # Update the display options in the datahub
        hub.model_instances[self.model_label]['display_options'] = (
            display_options)

    def preprocess(
            self,
            features):
        """
        Preprocess the feature vector.

        Parameters
        ----------
        features : ndarray
            Array of feature values.

        Returns
        -------
        ndarray
            Array of transformed feature values.
        """

        return self.preprocessor.transform(features)[0]

    def get_model(
            self,
            features,
            labels):
        """
        Get the machine learning model and its hyperparameters by reading \
        from the model folder path, the datahub, or by (re-)training.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : ndarray
            Values of the input labels.

        Returns
        -------
        object
            The object used to represent the prediction model.

        dict
            Dictionary with the values of the hyperparameters.
        """

        # Initialize the datahub
        hub = Datahub()

        # Check if the model files can be loaded from a folder path
        if self.model_folder_path:

            # Set the model file paths
            self.set_file_paths(self.model_folder_path)

            # Check if all file paths exist
            if all(exists(path) for path in (
                self.model_path, self.configuration_path,
                    self.hyperparameter_path)):

                # Set the update flag to False
                self.updated_model = False

                return (self.read_model_from_file(),
                        self.read_hyperparameters_from_file())

        # Else, check if the model files can be loaded from the datahub
        elif (self.model_label in hub.model_instances and compare_dictionaries(
                self.configuration,
                hub.model_instances[self.model_label]['configuration'])):

            # Set the update flag to False
            self.updated_model = False

            return (hub.model_instances[self.model_label]['prediction_model'],
                    hub.model_instances[self.model_label]['hyperparameters'])

        # Else, (re-)train the prediction model
        prediction_model, hyperparameters = self.train(features, labels)

        # Set the update flag to True
        self.updated_model = True

        return prediction_model, hyperparameters

    @abstractmethod
    def get_hyperparameter_set(
            self,
            proposal):
        """Get the hyperparameter set."""

    @abstractmethod
    def get_model_fit(
            self,
            features,
            labels,
            hyperparameters):
        """Get the machine learning model fit."""

    def tune_hyperparameters(
            self,
            features,
            labels):
        """
        Tune the hyperparameters of the machine learning model via sequential \
        model-based optimization using tree-structured Parzen estimators and \
        robust evaluation using stratified k-fold cross-validation.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : ndarray
            Values of the input labels.

        Returns
        -------
        tuned_hyperparameters : dict
            Dictionary with the values of the tuned hyperparameters.
        """

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the hyperparameter tuning
        hub.logger.display_info(
            'Performing Bayesian hyperparameter search for '
            f'"{self.model_label}" ...')

        def objective(proposal, trials, space):
            """Compute the objective function for a set of hyperparameters."""

            def compute_fold_score(indices):
                """Compute the score for a single train-validation split."""

                # Get the training and validation split
                split = [features[indices[0]], labels[indices[0]],
                         features[indices[1]], labels[indices[1]]]

                # Fit and transform the training and validation data
                split = [*preprocessor.fit_transform(split[0], split[1]),
                         *preprocessor.transform(split[2], split[3])]

                # Get the model fit
                prediction_model = self.get_model_fit(
                    split[0], split[1], hyperparameters)

                # Compute the training and validation scores
                scores = [-scorers[self.configuration['tune_score']](
                    labels, self.predict(features, prediction_model))
                    for features, labels in (split[:2], split[2:])]

                return max(scores)

            # Loop over the past trials
            for trial in trials:

                # Check if the trial has been accepted
                if trial['result']['status'] == STATUS_OK:

                    # Filter the trial values
                    values = {key: value[0] for key, value
                              in trial['misc']['vals'].items() if value}

                    # Check if the proposed set equals the trial set
                    if proposal == space_eval(space, values):

                        # Return an error status
                        return {'status': STATUS_FAIL}

            # Get the hyperparameter set
            hyperparameters = self.get_hyperparameter_set(proposal)

            # Compute the objective function value (score) across all folds
            fold_scores = map(compute_fold_score, (
                (training_indices, validation_indices)
                for training_indices, validation_indices in (
                    (where(self.configuration['tune_folds'] != number),
                     where(self.configuration['tune_folds'] == number))
                    for number in set(self.configuration['tune_folds']))))

            # Check if the first evaluation step has been passed
            if self.step > 0:

                # Log a message about the tuning status
                hub.logger.display_info(
                    f'Tuning hyperparameters for "{self.model_label}" '
                    f'({self.step}/{self.configuration["tune_evaluations"]}) '
                    f'- best loss: {min(filter(None, trials.losses()))} ...')

            # Increment the step variable
            self.step += 1

            return {'loss': max(fold_scores),
                    'params': hyperparameters,
                    'status': STATUS_OK}

        # Initialize the data preprocessor
        preprocessor = DataPreprocessor(self.preprocessing_steps, False)

        # Map the score labels to the score functions
        scorers = {'AUC': roc_auc_score, **score_loss_map}

        # Initialize the step variable
        self.step = 0

        # Generate a trials object for the evaluation history
        bayes_trials = Trials()

        # Run the hyperparameter tuning algorithm
        tuned_hyperparameters = fmin(
            fn=partial(objective, trials=bayes_trials, space=self.hp_space),
            space=self.hp_space, algo=tpe.suggest,
            max_evals=self.configuration['tune_evaluations'],
            trials=bayes_trials, return_argmin=False)

        # Log a message about the tuning status
        hub.logger.display_info(
            f'Completed hyperparameter tuning for "{self.model_label}" '
            f'({self.step}/{self.configuration["tune_evaluations"]}) '
            f'- best loss: {min(filter(None, bayes_trials.losses()))} ...')

        return tuned_hyperparameters

    def train(
            self,
            features,
            labels):
        """
        Train the machine learning model.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : ndarray
            Values of the input labels.

        Returns
        -------
        prediction_model : object
            The object used to represent the prediction model.

        hyperparameters : dict
            Dictionary with the values of the model hyperparameters.
        """

        # Log a message about the model fitting
        Datahub().logger.display_info(
            f'Fitting the model "{self.model_label}" to the data ...')

        # Get the hyperparameter set
        hyperparameters = self.get_hyperparameter_set(
            self.tune_hyperparameters(features, labels))

        # Get the model fit
        prediction_model = self.get_model_fit(
            self.preprocessed_features, self.preprocessed_labels,
            hyperparameters)

        return prediction_model, hyperparameters

    @abstractmethod
    def predict(
            self,
            features,
            predictor):
        """Predict the label values."""

    def predict_oof(
            self,
            features,
            labels):
        """
        Predict the out-of-folds (OOF) labels.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : ndarray
            Values of the input labels.

        Returns
        -------
        ndarray
            Array with the out-of-folds label predictions.
        """

        # Log a message about the out-of-folds prediction
        Datahub().logger.display_info(
            f'Performing {len(set(self.configuration["oof_folds"]))}-fold '
            'cross-validation to yield out-of-folds predictions for '
            f'"{self.model_label}" ...')

        def compute_fold_labels(indices):
            """Compute the out-of-folds labels for a single fold."""

            # Get the training and validation split
            split = [features[indices[0]], labels[indices[0]],
                     features[indices[1]]]

            # Fit and transform the training and validation data
            split = [*preprocessor.fit_transform(split[0], split[1]),
                     preprocessor.transform(split[2])[0]]

            # Get the model fit
            prediction_model = self.get_model_fit(
                split[0], split[1], self.hyperparameters)

            return (indices[1], self.predict(split[2], prediction_model))

        # Initialize the out-of-folds label prediction array
        oof_prediction = empty((len(labels),))

        # Initialize the data preprocessor
        preprocessor = DataPreprocessor(self.preprocessing_steps, False)

        # Compute the returns across all folds
        fold_returns = map(compute_fold_labels, (
            (training_indices, validation_indices)
            for training_indices, validation_indices in (
                (where(self.configuration['tune_folds'] != number),
                 where(self.configuration['tune_folds'] == number))
                for number in set(self.configuration['tune_folds']))))

        # Loop over the fold returns
        for fold_indices, fold_labels in fold_returns:

            # Insert the fold labels at the fold indices
            oof_prediction[fold_indices] = fold_labels

        return oof_prediction

    def inspect(
            self,
            features,
            labels,
            oof_folds):
        """
        Inspect the machine learning model.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : ndarray
            Values of the input labels.

        oof_folds : ndarray
            Out-of-fold split numbers.
        """

        # Check if the model should be first-time/repeatedly inspected
        if (self.model_label not in Datahub().model_inspections
                or self.updated_model):

            # Compute the model inspection results
            self.inspector.compute(
                self.prediction_model, self.hyperparameters, features, labels,
                self.preprocessing_steps, 30, oof_folds)

    def evaluate(
            self,
            features,
            labels):
        """
        Evaluate the machine learning model.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : ndarray
            Values of the input labels.
        """

        # Check if the model should be first-time/repeatedly evaluated
        if (self.model_label not in Datahub().model_evaluations
                or self.updated_model):

            # Compute the model evaluation results
            self.evaluator.compute(
                labels, (
                    self.predict(
                        self.preprocessed_features, self.prediction_model),
                    self.predict_oof(features, labels)))

    def set_file_paths(
            self,
            base_path):
        """
        Set the paths for model, configuration and hyperparameter files.

        Parameters
        ----------
        base_path : str
            Base path from which to access the model files.
        """

        # Set the file paths
        self.model_path, self.configuration_path, self.hyperparameter_path = (
            f'{base_path}/{filename}' for filename in (
                'model.sav', 'configuration.json', 'hyperparameters.json'))

    @abstractmethod
    def read_model_from_file(self):
        """Read the machine learning model from the model file path."""

    @abstractmethod
    def write_model_to_file(
            self,
            prediction_model):
        """Write the machine learning model to the model file path."""

    def read_configuration_from_file(self):
        """
        Read the configuration dictionary from the configuration file path.

        Returns
        -------
        dict
            Dictionary with information on the model configuration.
        """

        # Log a message about the configuration file reading
        Datahub().logger.display_info(
            f'Reading "{self.model_label}" configuration from file ...')

        # Open a file stream
        with open(self.configuration_path, 'r', encoding='utf-8') as file:

            # Load the configuration
            configuration = jload(file)

        # Loop over specific keys
        for key in ('feature_values', 'label_values', 'time_variable_values',
                    'tune_folds', 'oof_folds'):

            # Convert the value list into an array
            configuration[key] = array(configuration[key])

        return configuration

    def write_configuration_to_file(
            self,
            configuration):
        """
        Write the configuration dictionary to the configuration file path.

        Parameters
        ----------
        configuration : dict
            Dictionary with information on the model configuration.
        """

        # Loop over specific keys
        for key in ('feature_values', 'label_values', 'time_variable_values',
                    'tune_folds', 'oof_folds'):

            # Convert the array into a list
            configuration[key] = configuration[key].tolist()

        # Open a file stream
        with open(self.configuration_path, 'w', encoding='utf-8') as file:

            # Dump the dictionary to the file path
            jdump(configuration, file, sort_keys=False, indent=4)

    def read_hyperparameters_from_file(self):
        """
        Read the machine learning model hyperparameters from the \
        hyperparameter file path.

        Returns
        -------
        dict
            Dictionary with the values of the hyperparameters.
        """

        # Log a message about the parameter file reading
        Datahub().logger.display_info(
            f'Reading "{self.model_label}" hyperparameters from file ...')

        return jload(open(self.hyperparameter_path, 'r', encoding='utf-8'))

    def write_hyperparameters_to_file(
            self,
            hyperparameters):
        """
        Write the machine learning model hyperparameters to the \
        hyperparameter file path.

        Parameters
        ----------
        hyperparameters : dict
            Dictionary with the values of the hyperparameters.
        """

        # Open a file stream
        with open(self.hyperparameter_path, 'w', encoding='utf-8') as file:

            # Dump the dictionary to the file path
            jdump(hyperparameters, file, sort_keys=False, indent=4)
