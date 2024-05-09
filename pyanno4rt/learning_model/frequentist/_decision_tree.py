"""Decision tree outcome prediction model."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from json import dump as jdump
from json import load as jload
from os.path import exists
from pickle import dump, load

from functools import partial
from hyperopt import fmin, hp, space_eval, STATUS_FAIL, STATUS_OK, Trials, tpe
from numpy import empty
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model.preprocessing import DataPreprocessor
from pyanno4rt.learning_model.losses import brier_loss, log_loss
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
            oof_splits,
            display_options):

        # Initialize the datahub
        hub = Datahub()

        # Get the instance attributes from the arguments
        self.model_label = model_label
        self.model_folder_path = model_folder_path
        self.preprocessing_steps = preprocessing_steps

        # Initialize the file paths
        self.model_path = None
        self.configuration_path = None
        self.hyperparameter_path = None

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

        # Initialize the data preprocessor
        self.preprocessor = DataPreprocessor(preprocessing_steps)

        # Fit the data preprocessor and transform the input feature values
        self.preprocessed_features = self.preprocessor.fit_transform(
            dataset['feature_values'])

        # Initialize the boolean flag to indicate model updates
        self.updated_model = False

        # Get the decision tree model and its hyperparameters
        self.prediction_model, self.hyperparameters = (
            self.get_model(dataset['feature_values'], dataset['label_values']))

        # Check if the model has been updated or not yet registered
        if self.updated_model or self.model_label not in hub.model_instances:

            # Add the model instance to the datahub
            hub.model_instances[self.model_label] = {
                'prediction_model': self.prediction_model,
                'configuration': self.configuration,
                'hyperparameters': self.hyperparameters}

        # Check if the model should be inspected
        if inspect_model:

            # Inspect the model
            self.inspect(dataset['label_values'])

        # Check if the model should be evaluated
        if evaluate_model:

            # Evaluate the model
            self.evaluate(dataset['feature_values'], dataset['label_values'],
                          oof_splits)

        # Update the display options in the datahub
        hub.model_instances[self.model_label]['display_options'] = (
            display_options)

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

    def get_model(
            self,
            features,
            labels):
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

            # Check if all required files exists and if the configuration
            # dictionary equals the external configuration file content
            if (all(exists(path) for path in (
                    self.model_path, self.configuration_path,
                    self.hyperparameter_path)) and
                    self.configuration == self.read_configuration_from_file()):

                # Set the update flag to False
                self.updated_model = False

                return (self.read_model_from_file(),
                        self.read_hyperparameters_from_file())

        # Else, check if the model files can be loaded from the datahub
        else:

            # Initialize the datahub
            hub = Datahub()

            # Check if the model is already registered and if the configuration
            # dictionary equals the external configuration file content
            if (self.model_label in hub.model_instances and
                self.configuration == hub.model_instances[
                    self.model_label]['configuration']):

                # Set the update flag to False
                self.updated_model = False

                return (hub.model_instances[
                            self.model_label]['prediction_model'],
                        hub.model_instances[
                            self.model_label]['hyperparameters'])

        # Otherwise, train the outcome prediction model on the data
        prediction_model, hyperparameters = self.train(features, labels)

        # Set the update flag to True
        self.updated_model = True

        return prediction_model, hyperparameters

    def tune_hyperparameters_with_bayes(
            self,
            features,
            labels):
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

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the hyperparameter tuning
        hub.logger.display_info("Applying Bayesian hyperparameter tuning ...")

        def objective(proposal, trials, space):
            """Compute the objective function for a set of hyperparameters."""

            def compute_fold_score(indices):
                """Compute the score for a single train-validation split."""

                # Get the training and validation splits
                train_validate_split = [features[indices[0]],
                                        features[indices[1]],
                                        labels[indices[0]],
                                        labels[indices[1]]]

                # Fit the preprocessor and transform the training features
                train_validate_split[0] = preprocessor.fit_transform(
                    train_validate_split[0])

                # Transform the validation features
                train_validate_split[1] = preprocessor.transform(
                    train_validate_split[1])

                # Fit the model with the training split
                prediction_model.fit(train_validate_split[0],
                                     train_validate_split[2])

                # Get the training score
                training_score = -scorers[self.configuration['tune_score']](
                    train_validate_split[2], prediction_model.predict_proba(
                        train_validate_split[0])[:, 1])

                # Get the validation score
                validation_score = -scorers[self.configuration['tune_score']](
                    train_validate_split[3], prediction_model.predict_proba(
                        train_validate_split[1])[:, 1])

                return max(training_score, validation_score)

            # Check if the selected hyperparameter set has already been
            # evaluated from the past trials
            for trial in trials:

                # Check if the trial has been accepted
                if trial['result']['status'] == STATUS_OK:

                    # Reduce the assignments to the given values
                    reduced_values = {key: value[0]
                                      for key, value
                                      in trial['misc']['vals'].items()
                                      if value}

                    # Check if the proposed set equals the trial set
                    if proposal == space_eval(space, reduced_values):

                        # Return an error status
                        return {'status': STATUS_FAIL}

            # Create a dictionary with the model hyperparameters
            hyperparameters = {**proposal,
                               'random_state': 3,
                               'max_leaf_nodes': None,
                               'min_impurity_decrease': 0.0}

            # Initialize the model from the hyperparameter dictionary
            prediction_model = DecisionTreeClassifier(**hyperparameters)

            # Compute the objective function value (score) across all folds
            fold_scores = map(
                compute_fold_score, (
                    (training_indices, validation_indices)
                    for (training_indices, validation_indices)
                    in cross_validator.split(features, labels)))

            return {'loss': max(fold_scores),
                    'params': hyperparameters,
                    'status': STATUS_OK}

        # Initialize the stratified k-fold cross-validator
        cross_validator = StratifiedKFold(
            n_splits=self.configuration['tune_splits'], random_state=4,
            shuffle=True)

        # Initialize the data preprocessor
        preprocessor = DataPreprocessor(self.preprocessing_steps,
                                        verbose=False)

        # Map the score labels to the score functions
        scorers = {'Logloss': log_loss,
                   'Brier score': brier_loss,
                   'AUC': roc_auc_score}

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

        # Generate a trials object to store the evaluation history
        bayes_trials = Trials()

        # Run the optimization algorithm to get the tuned hyperparameters
        tuned_hyperparameters = fmin(fn=partial(objective, trials=bayes_trials,
                                                space=space),
                                     space=space,
                                     algo=tpe.suggest,
                                     max_evals=self.configuration[
                                         'tune_evaluations'],
                                     trials=bayes_trials,
                                     return_argmin=False)

        return tuned_hyperparameters

    def train(
            self,
            features,
            labels):
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
        tuned_hyperparameters = self.tune_hyperparameters_with_bayes(features,
                                                                     labels)

        # Create a dictionary with the model hyperparameters
        hyperparameters = {**tuned_hyperparameters,
                           'random_state': 3,
                           'max_leaf_nodes': None,
                           'min_impurity_decrease': 0.0}

        # Initialize the model from the hyperparameter dictionary
        prediction_model = DecisionTreeClassifier(**hyperparameters)

        # Fit the model with the preprocessed data
        prediction_model.fit(self.preprocessed_features, labels)

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
            features,
            labels,
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
        Datahub().logger.display_info(f"Performing {oof_splits}-fold "
                                      "cross-validation to yield "
                                      "out-of-folds predictions ...")

        def compute_fold_labels(indices):
            """Compute the out-of-folds labels for a single fold."""

            # Get the training and validation splits
            train_validate_split = [features[indices[0]],
                                    features[indices[1]],
                                    labels[indices[0]]]

            # Fit the preprocessor and transform the training features
            train_validate_split[0] = preprocessor.fit_transform(
                train_validate_split[0])

            # Transform the validation features
            train_validate_split[1] = preprocessor.transform(
                train_validate_split[1])

            # Initialize the model from the hyperparameter dictionary
            prediction_model = DecisionTreeClassifier(**self.hyperparameters)

            # Fit the model with the training split
            prediction_model.fit(train_validate_split[0],
                                 train_validate_split[2])

            return (indices[1], prediction_model.predict_proba(
                train_validate_split[1])[:, 1])

        # Initialize the out-of-folds label prediction array
        oof_prediction = empty((len(labels),))

        # Initialize the stratified k-fold cross-validator
        cross_validator = StratifiedKFold(
            n_splits=oof_splits, random_state=4, shuffle=True)

        # Initialize the data preprocessor
        preprocessor = DataPreprocessor(self.preprocessing_steps,
                                        verbose=False)

        # Compute the returns (indices and labels) across all folds
        fold_returns = map(
            compute_fold_labels, (
                (training_indices, validation_indices)
                for (training_indices, validation_indices)
                in cross_validator.split(features, labels)))

        # Loop over the fold returns
        for fold_indices, fold_labels in fold_returns:

            # Insert the fold labels at the fold indices
            oof_prediction[fold_indices] = fold_labels

        return oof_prediction

    def inspect(
            self,
            labels):
        """."""

        # Initialize the datahub
        hub = Datahub()

        # Check if the model should be first-time/repeatedly inspected
        if (self.model_label not in (*hub.model_inspections,)
                or self.updated_model):

            # Initialize the model inspector
            inspector = ModelInspector(
                model_name=self.model_label,
                model_class='Decision Tree')

            # Run the model inspections
            inspector.compute(model=self.prediction_model,
                              features=self.preprocessed_features,
                              labels=labels,
                              number_of_repeats=30)

            # Add the model inspector to the datahub
            hub.model_inspections[self.model_label] = inspector

    def evaluate(
            self,
            features,
            labels,
            oof_splits):
        """."""

        # Initialize the datahub
        hub = Datahub()

        # Check if the model should be first-time/repeatedly evaluated
        if (self.model_label not in (*hub.model_evaluations,)
                or self.updated_model):

            # Initialize the model evaluator
            evaluator = ModelEvaluator(
                model_name=self.model_label,
                true_labels=labels)

            # Run the model evaluations on training and OOF predictions
            evaluator.compute(predicted_labels=(
                self.predict(self.preprocessed_features),
                self.predict_oof(features, labels, oof_splits)))

            # Add the model evaluator to the datahub
            hub.model_evaluations[self.model_label] = evaluator

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

        # Set the model, configuration and hyperparameter file paths
        (self.model_path, self.configuration_path,
         self.hyperparameter_path) = (''.join((base_path, '/', filename))
                                      for filename in ('model.sav',
                                                       'configuration.json',
                                                       'hyperparameters.json'))

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
        Datahub().logger.display_info(f"Reading '{self.model_label}' model "
                                      "from file ...")

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
        with open(self.model_path, 'wb') as file:
            dump(prediction_model, file)

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
        Datahub().logger.display_info(f"Reading '{self.model_label}' "
                                      "configuration from file ...")

        return jload(open(self.configuration_path, 'r', encoding='utf-8'))

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
        with open(self.configuration_path, 'w', encoding='utf-8') as file:
            jdump(configuration, file, sort_keys=False, indent=4)

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
        Datahub().logger.display_info(f"Reading '{self.model_label}' "
                                      "hyperparameters from file ...")

        return jload(open(self.hyperparameter_path, 'r', encoding='utf-8'))

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
        with open(self.hyperparameter_path, 'w', encoding='utf-8') as file:
            jdump(hyperparameters, file, sort_keys=False, indent=4)
