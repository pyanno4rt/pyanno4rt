"""Decision tree outcome prediction model."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from json import dump as jdump
from json import load as jload
from os.path import exists
from pickle import dump, load
from statistics import mean

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from hyperopt import fmin, hp, space_eval, STATUS_FAIL, STATUS_OK, Trials, tpe
from numpy import empty
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model.preprocessing import DataPreprocessor
from pyanno4rt.learning_model.losses import log_loss
from pyanno4rt.learning_model.inspection import ModelInspector
from pyanno4rt.learning_model.evaluation import ModelEvaluator

# %% Class definition


class DecisionTreeModel():
    """
    Decision tree outcome prediction model class.

    This class enables building an individual preprocessing pipeline, fit the \
    decision tree model from the input data, inspect the model, make \
    predictions with the model, and assess the predictive performance using \
    multiple evaluation metrics.

    The training process includes sequential model-based hyperparameter \
    optimization with tree-structured Parzen estimators and stratified k-fold \
    cross-validation for the objective function evaluation. Cross-validation \
    is also applied to (optionally) inspect the validation feature \
    importances and to generate out-of-folds predictions as a full \
    reconstruction of the input labels for generalization assessment.

    Parameters
    ----------
    model_label : string
        Label for the decision tree model to be used for file naming.

    dataset : dict
        Dictionary with the raw data set, the label viewpoint, the label \
        bounds, the feature values and names, and the label values and names \
        after modulation. In a compact way, this represents the input data \
        for the decision tree model.

    preprocessing_steps : tuple
        Sequence of labels associated with preprocessing algorithms which \
        make up the preprocessing pipeline for the decision tree model. \
        Current available algorithm labels are:

        - transformers : 'Equalizer', 'StandardScaler', 'Whitening'.

    tune_space : dict
        Search space for the Bayesian hyperparameter optimization, including

        - 'criterion' : measure for the quality of a split;
        - 'splitter' : splitting strategy at each node;
        - 'max_depth' : maximum depth of the tree;
        - 'min_samples_split' : minimum number of samples required for \
            splitting each node;
        - 'min_samples_leaf' : minimum number of samples required at each node;
        - 'min_weight_fraction_leaf' : minimum weighted fraction of the \
            weights sum required at each node;
        - 'max_features' : maximum number of features taken into account when \
            looking for the best split at each node;
        - 'class_weight' : weights associated with the classes;
        - 'ccp_alpha' : complexity parameter for minimal cost-complexity \
            pruning.

    tune_evaluations : int
        Number of evaluation steps (trials) for the Bayesian hyperparameter \
        optimization.

    tune_score : string
        Scoring function for the evaluation of the hyperparameter set \
        candidates. Current available scorers are:

        - 'log_loss' : negative log-likelihood score;
        - 'roc_auc_score' : area under the ROC curve score.

    tune_splits : int
        Number of splits for the stratified cross-validation within each \
        hyperparameter optimization step.

    inspect_model : bool
        Indicator for the inspection of the model, e.g. the feature \
        importances.

    evaluate_model : bool
        Indicator for the evaluation of the model, e.g. the model KPIs.

    oof_splits : int
        Number of splits for the stratified cross-validation within the \
        out-of-folds evaluation step of the decision tree model.

    Attributes
    ----------
    preprocessor : object of class `DataPreprocessor`
        Instance of the class `DataPreprocessor`, which holds methods to \
        build the preprocessing pipeline, fit with the input features, \
        transform the features, and derive the gradient of the preprocessing \
        algorithms w.r.t the features.

    features : ndarray
        Values of the input features.

    labels : ndarray
        Values of the input labels.

    configuration : dict
        Dictionary with information for the modeling, i.e., the dataset, the \
        preprocessing, and the hyperparameter search space.

    model_path : string
        Path for storing and retrieving the decision tree model.

    configuration_path : string
        Path for storing and retrieving the configuration dictionary.

    hyperparameter_path : string
        Path for storing and retrieving the hyperparameter dictionary.

    updated_model : bool
        Indicator for the update status of the model, triggers recalculating \
        the model inspection and model evaluation classes.

    prediction_model : object of class `DecisionTreeClassifier`
        Instance of the class `DecisionTreeClassifier`, which holds methods \
        to make predictions from the decision tree model.

    inspector : object of class `ModelInspector`
        Instance of the class `ModelInspector`, which holds methods to \
        compute model inspection values, e.g. feature importances.

    training_prediction : ndarray
        Array with the label predictions on the input data.

    oof_prediction : ndarray
        Array with the out-of-folds predictions on the input data.

    evaluator : object of class `ModelEvaluator`
        Instance of the class `ModelEvaluator`, which holds methods to \
        compute the evaluation metrics for a given array with label \
        predictions.

    Notes
    -----
    Currently, the preprocessing pipeline for the model is restricted to \
    transformations of the input feature values, e.g. scaling, dimensionality \
    reduction or feature engineering. Transformations which affect the input \
    labels in the same way, e.g. resampling or outlier removal, are not yet \
    possible.
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
            tune_splits,
            inspect_model,
            evaluate_model,
            oof_splits):

        # Initialize the datahub
        hub = Datahub()

        # Get the information units from the datahub
        model_instances = hub.model_instances
        model_inspections = hub.model_inspections
        model_evaluations = hub.model_evaluations

        # Get the model label from the argument
        self.model_label = model_label

        # Get the model folder path from the argument
        self.model_folder_path = model_folder_path

        # Initialize the data preprocessor
        self.preprocessor = DataPreprocessor(preprocessing_steps)

        # Fit the data preprocessor to the input feature values
        self.preprocessor.fit(dataset['feature_values'])

        # Transform the input feature values with the preprocessor
        self.features = self.preprocess(dataset['feature_values'])

        # Get the input label values from the dataset
        self.labels = dataset['label_values']

        # Create the configuration dictionary with the modeling information
        self.configuration = {'data': [dataset['feature_values'].shape[0],
                                       dataset['feature_names'],
                                       dataset['label_values'].shape[0],
                                       dataset['label_names']],
                              'label_viewpoint': dataset['label_viewpoint'],
                              'label_bounds': dataset['label_bounds'],
                              'preprocessing_steps': preprocessing_steps,
                              'tune_space': {
                                  'criterion': tune_space.get(
                                      'criterion', ['gini', 'entropy']),
                                  'splitter': tune_space.get(
                                      'splitter', ['best', 'random']),
                                  'max_depth': tune_space.get(
                                      'max_depth', list(range(1, 21))),
                                  'min_samples_split': tune_space.get(
                                      'min_samples_split', [0.0, 1.0]),
                                  'min_samples_leaf': tune_space.get(
                                      'min_samples_leaf', [0.0, 0.5]),
                                  'min_weight_fraction_leaf': tune_space.get(
                                      'min_weight_fraction_leaf', [0.0, 0.5]),
                                  'max_features': tune_space.get(
                                      'max_features', list(range(
                                          1, self.features.shape[1]+1))),
                                  'class_weight': tune_space.get(
                                      'class_weight', [None, 'balanced']),
                                  'ccp_alpha': tune_space.get(
                                      'ccp_alpha', [0.0, 1.0])},
                              'tune_evaluations': tune_evaluations,
                              'tune_score': tune_score,
                              'tune_splits': tune_splits}

        # Initialize the boolean flag to indicate model updates
        self.updated_model = False

        # Get the decision tree model and its hyperparameters
        self.prediction_model, self.hyperparameters = (
            self.get_prediction_model())

        # Check if the model has been updated or is unregistered in the datahub
        if self.updated_model or self.model_label not in model_instances:

            # Add the model instance to the datahub
            model_instances[self.model_label] = {
                'prediction_model': self.prediction_model,
                'configuration': self.configuration,
                'hyperparameters': self.hyperparameters}

        # Check if the model should be first-time/repeatedly inspected
        if (inspect_model and (self.updated_model or self.model_label
                               not in (*model_inspections,))):

            # Initialize the model inspector
            self.inspector = ModelInspector(
                model_name=self.model_label,
                model_class='Decision Tree')

            # Run the model inspections
            self.inspector.compute(model=self.prediction_model,
                                   features=self.features,
                                   labels=self.labels,
                                   number_of_repeats=30)

            # Add the model inspector to the datahub
            model_inspections[self.model_label] = self.inspector

        # Else, check if the inspection should be retrieved from the datahub
        elif inspect_model:

            # Log a message about the inspector retrieval
            hub.logger.display_info("Retrieving model inspector from datahub "
                                    "...")

            # Get the model inspector from the datahub
            self.inspector = model_inspections[self.model_label]

        # Check if the model should be first-time/repeatedly evaluated
        if (evaluate_model and (self.updated_model or self.model_label
                                not in (*model_evaluations,))):

            # Predict the training and OOF labels
            self.training_prediction = self.predict(self.features)
            self.oof_prediction = self.predict_oof(oof_splits)

            # Initialize the model evaluator
            self.evaluator = ModelEvaluator(
                model_name=self.model_label,
                true_labels=self.labels)

            # Run the model evaluations
            self.evaluator.compute(predicted_labels=(self.training_prediction,
                                                     self.oof_prediction))

            # Add the model evaluator to the datahub
            model_evaluations[self.model_label] = self.evaluator

        # Else, check if the evaluation should be retrieved from the datahub
        elif evaluate_model:

            # Log a message about the evaluator retrieval
            hub.logger.display_info("Retrieving model evaluator from datahub "
                                    "...")

            # Get the model evaluator from the datahub
            self.evaluator = model_evaluations[self.model_label]

    def set_file_paths(
            self,
            base_path):
        """
        Set the paths for model, configuration and hyperparameter files.

        Parameters
        ----------
        base_path : string
            Base path from which to access the model files.
        """
        # Set the specific model, configuration and hyperparameter file paths
        self.model_path = ''.join((base_path, '/', 'Model.sav'))
        self.configuration_path = ''.join((base_path, '/',
                                           'Configuration.json'))
        self.hyperparameter_path = ''.join((base_path, '/',
                                            'Hyperparameters.json'))

    def preprocess(
            self,
            features):
        """
        Preprocess the input feature vector with the built pipeline.

        Parameters
        ----------
        features : ndarray
            Array of input feature values.

        Returns
        -------
        ndarray
            Array of transformed feature values.
        """
        return self.preprocessor.transform(features)

    def get_prediction_model(self):
        """
        Get the decision tree outcome prediction model by reading from the \
        model file path, the datahub, or by training.

        Returns
        -------
        object of class `DecisionTreeClassifier`
            Instance of the class `DecisionTreeClassifier`, which holds \
            methods to make predictions from the decision tree model.
        """
        # Check if the model files can be loaded from the model folder path
        if self.model_folder_path:

            # Set the base path to the model folder path
            self.set_file_paths(self.model_folder_path)

            if all(exists(path) for path in (
                    self.model_path, self.configuration_path,
                    self.hyperparameter_path)):

                # Check if the configuration dictionary of the class equals the
                # content of the external configuration file
                if self.configuration == self.read_configuration_from_file():

                    # Set the update flag to False
                    self.updated_model = False

                    return (self.read_model_from_file(),
                            self.read_hyperparameters_from_file())

        # Else, check if the model files can be loaded from the datahub
        else:

            # Initialize the datahub
            hub = Datahub()

            # Check if the model is stored in the datahub
            if self.model_label in hub.model_instances:

                # Check if the configuration dictionary of the class equals the
                # content of the configuration dictionary in the datahub
                if self.configuration == hub.model_instances[
                        self.model_label]['configuration']:

                    # Set the update flag to False
                    self.updated_model = False

                    return (
                        hub.model_instances[
                            self.model_label]['prediction_model'],
                        hub.model_instances[
                            self.model_label]['hyperparameters'])

        # Otherwise, fit a new model
        prediction_model, hyperparameters = self.train()

        # Set the update flag to True
        self.updated_model = True

        return prediction_model, hyperparameters

    def read_model_from_file(self):
        """
        Read the decision tree outcome prediction model from the model file \
        path.

        Returns
        -------
        object of class `DecisionTreeClassifier`
            Instance of the class `DecisionTreeClassifier`, which holds \
            methods to make predictions from the decision tree model.
        """
        # Log a message about the model file reading
        Datahub().logger.display_info("Reading '{}' model from file ..."
                                      .format(self.model_label))

        return load(open(self.model_path, 'rb'))

    def write_model_to_file(
            self,
            prediction_model):
        """
        Write the decision tree outcome prediction model to the model file \
        path.

        Parameters
        ----------
        prediction_model : object of class `DecisionTreeClassifier`
            Instance of the class `DecisionTreeClassifier`, which holds \
            methods to make predictions from the decision tree model.
        """
        # Dump the model to the model file path
        dump(prediction_model, open(self.model_path, 'wb'))

    def read_configuration_from_file(self):
        """
        Read the configuration dictionary from the configuration file path.

        Returns
        -------
        dict
            Dictionary with information for the modeling, i.e., the dataset, \
            the preprocessing steps, and the hyperparameter search space.
        """
        # Log a message about the configuration file reading
        Datahub().logger.display_info("Reading '{}' configuration from file "
                                      "..."
                                      .format(self.model_label))

        return jload(open(self.configuration_path, 'r'))

    def write_configuration_to_file(
            self,
            configuration):
        """
        Write the configuration dictionary to the configuration file path.

        Parameters
        ----------
        configuration : dict
            Dictionary with information for the modeling, i.e., the dataset, \
            the preprocessing steps, and the hyperparameter search space.
        """
        # Dump the configuration dictionary to the configuration file path
        jdump(configuration, open(self.configuration_path, 'w'),
              sort_keys=False, indent=4)

    def read_hyperparameters_from_file(self):
        """
        Read the decision tree outcome prediction model hyperparameters from \
        the hyperparameter file path.

        Returns
        -------
        dict
            Dictionary with the hyperparameter names and values for the \
            decision tree outcome prediction model.
        """
        # Log a message about the parameter file reading
        Datahub().logger.display_info("Reading '{}' hyperparameters from file "
                                      "..."
                                      .format(self.model_label))

        return jload(open(self.hyperparameter_path, 'r'))

    def write_hyperparameters_to_file(
            self,
            hyperparameters):
        """
        Write the hyperparameter dictionary to the hyperparameter file path.

        Parameters
        ----------
        hyperparameters : dict
            Dictionary with the hyperparameter names and values for the \
            decision tree outcome prediction model.
        """
        # Dump the hyperparameter dictionary to the hyperparameter file path
        jdump(hyperparameters, open(self.hyperparameter_path, 'w'),
              sort_keys=False, indent=4)

    def tune_hyperparameters_with_bayes(self):
        """
        Tune the hyperparameters of the decision tree model via sequential \
        model-based optimization using the tree-structured Parzen estimator. \
        As a variation, the objective function is evaluated based on a \
        stratified k-fold cross-validation.

        Returns
        -------
        tuned_hyperparameters : dict
            Dictionary with the hyperparameter names and values tuned via \
            Bayesian hyperparameter optimization.
        """
        # Log a message about the hyperparameter tuning
        Datahub().logger.display_info("Applying Bayesian hyperparameter "
                                      "tuning ...")

        def objective(proposal, trials, space):
            """Compute the objective function for a set of hyperparameters."""

            def compute_fold_score(indices):
                """Compute the score for a single train-validation split."""
                # Get the training and validation indices from the argument
                training_indices = indices[0]
                validation_indices = indices[1]

                # Get the training and validation splits
                train_validation_split = (self.features[training_indices],
                                          self.features[validation_indices],
                                          self.labels[training_indices],
                                          self.labels[validation_indices])

                # Fit the model with the training split
                prediction_model.fit(train_validation_split[0],
                                     train_validation_split[2])

                # Predict the labels for the validation split
                predicted_labels = prediction_model.predict_proba(
                        train_validation_split[1])[:, 1]

                return -scores[self.configuration['tune_score']](
                    train_validation_split[3], predicted_labels)

            # Check if the selected hyperparameter set has already been \
            # evaluated from the past trials
            for trial in trials:

                # Check if the trial has been accepted
                if trial['result']['status'] == STATUS_OK:

                    # Get the hyperparameter assignments from the trial
                    values = trial['misc']['vals']

                    # Reduce the assignments to the given values
                    reduced_values = {key: value[0]
                                      for key, value in values.items()
                                      if value}

                    # Return a failure if the proposed set equals the trial set
                    if proposal == space_eval(space, reduced_values):
                        return {'status': STATUS_FAIL}

            # Create a dictionary with the model hyperparameters
            hyperparameters = {**proposal,
                               'random_state': 3,
                               'max_leaf_nodes': None,
                               'min_impurity_decrease': 0.0}

            # Initialize the model from the hyperparameter dictionary
            prediction_model = DecisionTreeClassifier(**hyperparameters)

            # Compute the objective function value (loss) across all folds
            losses = map(
                compute_fold_score,
                ((train_indices, validation_indices)
                 for (train_indices, validation_indices)
                 in cross_validator.split(self.features, self.labels)))

            return {'loss': mean(losses), 'params': hyperparameters,
                    'status': STATUS_OK}

        # Initialize the stratified k-fold cross-validator
        cross_validator = StratifiedKFold(
            n_splits=self.configuration['tune_splits'], random_state=4,
            shuffle=True)

        # Map the score labels to the score functions
        scores = {'log_loss': log_loss,
                  'roc_auc_score': roc_auc_score}

        # Define the search space for the hyperparameters
        space = {
            'criterion': hp.choice(
                'criterion', self.configuration['tune_space']['criterion']),
            'splitter': hp.choice(
                'splitter', self.configuration['tune_space']['splitter']),
            'max_depth': hp.choice(
                'max_depth', self.configuration['tune_space']['max_depth']),
            'min_samples_split': hp.uniform(
                'min_samples_split',
                self.configuration['tune_space']['min_samples_split'][0],
                self.configuration['tune_space']['min_samples_split'][1]),
            'min_samples_leaf': hp.uniform(
                'min_samples_leaf',
                self.configuration['tune_space']['min_samples_leaf'][0],
                self.configuration['tune_space']['min_samples_leaf'][1]),
            'min_weight_fraction_leaf': hp.uniform(
                'min_weight_fraction_leaf',
                self.configuration[
                    'tune_space']['min_weight_fraction_leaf'][0],
                self.configuration[
                    'tune_space']['min_weight_fraction_leaf'][1]),
            'max_features': hp.choice(
                'max_features', self.configuration[
                    'tune_space']['max_features']),
            'class_weight': hp.choice(
                'class_weight', self.configuration[
                    'tune_space']['class_weight']),
            'ccp_alpha': hp.uniform(
                'ccp_alpha', self.configuration['tune_space']['ccp_alpha'][0],
                self.configuration['tune_space']['ccp_alpha'][1])
            }

        # State the algorithm for the sequential search
        tpe_algorithm = tpe.suggest

        # Generate a trials object to store the evaluation history
        bayes_trials = Trials()

        # Run the optimization algorithm to get the tuned hyperparameters
        tuned_hyperparameters = fmin(fn=partial(objective, trials=bayes_trials,
                                                space=space),
                                     space=space,
                                     algo=tpe_algorithm,
                                     max_evals=self.configuration[
                                         'tune_evaluations'],
                                     trials=bayes_trials,
                                     return_argmin=False)

        return tuned_hyperparameters

    def train(self):
        """
        Train the decision tree outcome prediction model.

        Returns
        -------
        prediction_model : object of class `DecisionTreeClassifier`
            Instance of the class `DecisionTreeClassifier`, which holds \
            methods to make predictions from the decision tree model.
        """
        # Log a message about the model fitting
        Datahub().logger.display_info("Fitting the model to the data ...")

        # Get the tuned hyperparameters from the Bayesian optimization
        tuned_hyperparameters = self.tune_hyperparameters_with_bayes()

        # Create a dictionary with the model hyperparameters
        hyperparameters = {**tuned_hyperparameters,
                           'random_state': 3,
                           'max_leaf_nodes': None,
                           'min_impurity_decrease': 0.0}

        # Initialize the model from the hyperparameter dictionary
        prediction_model = DecisionTreeClassifier(**hyperparameters)

        # Fit the model with the data
        prediction_model.fit(self.features, self.labels)

        return prediction_model, hyperparameters

    def predict(
            self,
            features):
        """
        Predict the label values from the feature values.

        Parameters
        ----------
        features : ndarray
            Array of input feature values.

        Returns
        -------
        float or ndarray
            Floating-point label prediction or array of label predictions.
        """
        # Check if the feature array has only a single row
        if features.shape[0] == 1:

            # Return a single label prediction value
            return self.prediction_model.predict_proba(features)[0][1]

        # Otherwise, return an array with label predictions
        return self.prediction_model.predict_proba(features)[:, 1]

    def predict_oof(
            self,
            oof_splits):
        """
        Predict the out-of-folds (OOF) labels using a stratified k-fold \
        cross-validation.

        Parameters
        ----------
        oof_splits : int
            Number of splits for the stratified cross-validation.

        Returns
        -------
        ndarray
            Array with the out-of-folds label predictions.
        """
        # Log a message about the out-of-folds prediction
        Datahub().logger.display_info("Performing {}-fold cross-validation to "
                                      "yield out-of-folds predictions ..."
                                      .format(str(oof_splits)))

        def compute_fold_labels(indices):
            """Compute the out-of-folds labels for a single fold."""
            # Get the function parameters from the arguments
            training_indices = indices[0]
            validation_indices = indices[1]

            # Get the training and validation splits
            train_validation_split = (self.features[training_indices],
                                      self.features[validation_indices],
                                      self.labels[training_indices])

            # Initialize the model from the hyperparameter dictionary
            prediction_model = DecisionTreeClassifier(**self.hyperparameters)

            # Fit the model with the training split
            prediction_model.fit(train_validation_split[0],
                                 train_validation_split[2])

            # Predict the labels for the validation split
            oof_fold_labels = prediction_model.predict_proba(
                train_validation_split[1])[:, 1]

            return validation_indices, oof_fold_labels

        # Initialize the out-of-folds label prediction array
        oof_prediction = empty((len(self.labels),))

        # Initialize the stratified k-fold cross-validator
        cross_validator = StratifiedKFold(n_splits=oof_splits, random_state=4,
                                          shuffle=True)

        # Compute the out-of-folds labels across all folds
        oof_results = ThreadPoolExecutor().map(
            compute_fold_labels,
            ((train_indices, validation_indices)
             for (train_indices, validation_indices)
             in cross_validator.split(self.features, self.labels)))

        for result in oof_results:
            oof_prediction[result[0]] = result[1]

        return oof_prediction
