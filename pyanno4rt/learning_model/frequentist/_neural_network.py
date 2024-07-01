"""Neural network model."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from json import dump as jdump, load as jload
from os.path import exists

from functools import partial
from h5py import File
from hyperopt import fmin, hp, space_eval, STATUS_FAIL, STATUS_OK, Trials, tpe
from numpy import array, empty, where
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model.evaluation import ModelEvaluator
from pyanno4rt.learning_model.frequentist.addons import (
    build_iocnn, build_standard_nn, loss_map, optimizer_map)
from pyanno4rt.learning_model.inspection import ModelInspector
from pyanno4rt.learning_model.losses import loss_map as score_loss_map
from pyanno4rt.learning_model.preprocessing import DataPreprocessor
from pyanno4rt.tools import compare_dictionaries

# %% Class definition


class NeuralNetworkModel():
    """
    Neural network model class.

    This class enables building an individual preprocessing pipeline, fit the \
    neural network model from the input data, inspect the model, make \
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
        Label for the neural network model to be used for file naming.

    dataset : dict
        Dictionary with the raw data set, the label viewpoint, the label \
        bounds, the feature values and names, and the label values and names \
        after modulation. In a compact way, this represents the input data \
        for the neural network model.

    preprocessing_steps : tuple
        Sequence of labels associated with preprocessing algorithms which \
        make up the preprocessing pipeline for the neural network model. \
        Current available algorithm labels are:

        - transformers : 'Equalizer', 'StandardScaler', 'Whitening'.

    architecture : {'input-convex', 'standard'}
        Type of architecture for the neural network model. Current available \
        architectures are:

        - 'input-convex' : builds the input-convex network architecture;
        - 'standard' : builds the standard feed-forward network architecture.

    max_hidden_layers : int
        Maximum number of hidden layers for the neural network model.

    tune_space : dict
        Search space for the Bayesian hyperparameter optimization, including

        - 'input_neuron_number' : number of neurons for the input layer;
        - 'input_activation' : activation function for the input layer \
           ('elu', 'exponential', 'gelu', 'linear', 'leaky_relu', 'relu', \
            'softmax', 'softplus', 'swish');
        - 'hidden_neuron_number' : number of neurons for the hidden layer(s);
        - 'hidden_activation' : activation function for the hidden layer(s) \
           ('elu', 'gelu', 'linear', 'leaky_relu', 'relu', 'softmax', \
            'softplus', 'swish');
        - 'input_dropout_rate' : dropout rate for the input layer;
        - 'hidden_dropout_rate' : dropout rate for the hidden layer(s);
        - 'batch_size' : batch size;
        - 'learning_rate' : learning rate
        - 'optimizer' : algorithm for the optimization of the network \
           ('Adam', 'Ftrl', 'SGD');
        - 'loss' : loss function for the optimization of the network \
           ('BCE', 'FocalBCE', 'KLD').

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

    inspect_model : bool
        Indicator for the inspection of the model, e.g. the feature \
        importances.

    evaluate_model : bool
        Indicator for the evaluation of the model, e.g. the model KPIs.

    oof_splits : int
        Number of splits for the stratified cross-validation within the \
        out-of-folds evaluation step of the logistic regression model.

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
        Path for storing and retrieving the neural network model.

    configuration_path : string
        Path for storing and retrieving the configuration dictionary.

    hyperparameter_path : string
        Path for storing and retrieving the hyperparameter dictionary.

    updated_model : bool
        Indicator for the update status of the model, triggers recalculating \
        the model inspection and model evaluation classes.

    prediction_model : object of class `Functional`
        Instance of the class `Functional`, which holds methods to make \
        predictions from the neural network model.

    optimization_model : object of class `Functional`
        Instance of the class `Functional`, equivalent to \
        ``prediction_model``, but skips the sigmoid output activation.

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
            architecture,
            max_hidden_layers,
            tune_space,
            tune_evaluations,
            tune_score,
            inspect_model,
            evaluate_model,
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
            'tune_space': {
                'input_neuron_number': tune_space.get(
                    'input_neuron_number', [2**x for x in range(1, 12)]),
                'input_activation': tune_space.get(
                    'input_activation', [
                        'elu', 'gelu', 'leaky_relu', 'linear', 'relu',
                        'softmax', 'softplus', 'swish']),
                'hidden_neuron_number': tune_space.get(
                    'hidden_neuron_number', [2**x for x in range(1, 12)]),
                'hidden_activation': tune_space.get(
                    'hidden_activation', [
                        'elu', 'gelu', 'leaky_relu', 'linear', 'relu',
                        'softmax', 'softplus', 'swish']),
                'input_dropout_rate': tune_space.get(
                    'input_dropout_rate', [0.0, 0.1, 0.25, 0.5, 0.75]),
                'hidden_dropout_rate': tune_space.get(
                    'input_dropout_rate', [0.0, 0.1, 0.25, 0.5, 0.75]),
                'batch_size': tune_space.get('batch_size', [4, 8, 16, 32]),
                'learning_rate': tune_space.get('learning_rate', [1e-5, 1e-2]),
                'optimizer': tune_space.get(
                    'optimizer', ['Adam', 'Ftrl', 'SGD']),
                'loss': tune_space.get('loss', ['BCE', 'FocalBCE', 'KLD'])},
            'tune_evaluations': tune_evaluations,
            'tune_score': tune_score}

        # Initialize the data preprocessor
        self.preprocessor = DataPreprocessor(preprocessing_steps)

        # Fit the data preprocessor and transform the input feature values
        self.preprocessed_features, self.preprocessed_labels = (
            self.preprocessor.fit_transform(
                dataset['feature_values'], dataset['label_values']))

        # Initialize the boolean flag to indicate model updates
        self.updated_model = False

        # Get the logistic regression model and its hyperparameters
        self.prediction_model, self.hyperparameters = (
            self.get_prediction_model(
                dataset['feature_values'], dataset['label_values']))

        # Get the optimizable neural network model
        self.optimization_model = self.get_optimization_model(
            dataset['feature_values'], dataset['label_values'])

        # Check if the model has been updated or not yet registered
        if self.updated_model or self.model_label not in hub.model_instances:

            # Add the model instance to the datahub
            hub.model_instances[self.model_label] = {
                'prediction_model': self.prediction_model,
                'optimization_model': self.optimization_model,
                'configuration': self.configuration,
                'hyperparameters': self.hyperparameters}

        # Check if the model should be inspected
        if inspect_model:

            # Initialize the model inspector
            self.inspector = ModelInspector(model_label)

            # Inspect the model
            self.inspect(
                dataset['feature_values'], dataset['label_values'],
                dataset['oof_folds'])

        # Check if the model should be evaluated
        if evaluate_model:

            # Initialize the model evaluator
            self.evaluator = ModelEvaluator(model_label)

            # Evaluate the model
            self.evaluate(
                dataset['feature_values'], dataset['label_values'])

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

        return self.preprocessor.transform(features)[0]

    def get_prediction_model(
            self,
            features,
            labels):
        """
        Get the neural network outcome prediction model by reading from the \
        model file path, the datahub, or by training.

        Returns
        -------
        object of class `Functional`
            Instance of the class `Functional`, which holds methods to make \
            predictions from the neural network model.
        """

        # Check if the model files can be loaded from the model folder path
        if self.model_folder_path:

            # Set the base path to the model folder path
            self.set_file_paths(self.model_folder_path)

            # Check if all required files exists and if the configuration
            # dictionary equals the external configuration file content
            if all(exists(path) for path in (
                    self.model_path, self.configuration_path,
                    self.hyperparameter_path)):

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
            if (self.model_label in hub.model_instances
                and compare_dictionaries(
                    self.configuration, hub.model_instances[
                        self.model_label]['configuration'])):

                # Set the update flag to False
                self.updated_model = False

                return (
                    hub.model_instances[self.model_label]['prediction_model'],
                    hub.model_instances[self.model_label]['hyperparameters']
                    )

        # Otherwise, train the outcome prediction model on the data
        prediction_model, hyperparameters = self.train(features, labels)

        # Set the update flag to True
        self.updated_model = True

        return prediction_model, hyperparameters

    def get_optimization_model(
            self,
            features,
            labels):
        """
        Get the neural network outcome optimization model.

        Returns
        -------
        object of class `Functional`
            Instance of the class `Functional`, which holds methods to make \
            predictions from the neural network model.
        """

        # Log a message about the optimization model retrieval
        Datahub().logger.display_info("Getting the optimization model ...")

        # Build the network architecture
        optimization_model = self.build_network(
            features.shape[1], labels.ndim, self.hyperparameters,
            squash_output=False)

        # Compile the model
        optimization_model.compile(
            optimizer=optimizer_map[self.hyperparameters['optimizer']](
                learning_rate=self.hyperparameters['learning_rate']),
            loss=loss_map[self.hyperparameters['loss']])

        # Set the network weights for the model
        optimization_model.set_weights(self.prediction_model.get_weights())

        return optimization_model

    def build_network(
            self,
            input_shape,
            output_shape,
            hyperparameters,
            squash_output):
        """
        Build the neural network architecture with the functional API.

        Parameters
        ----------
        input_shape : int
            Shape of the input features.

        output_shape : int
            Shape of the output labels.

        hyperparameters : dict
            Dictionary with the hyperparameter names and values for the \
            neural network outcome prediction model.

        squash_output : bool
            Indicator for the use of a sigmoid activation function in the \
            output layer.

        Returns
        -------
        object of class 'Functional'
            Instance of the class `Functional`, which holds methods to make \
            predictions from the neural network model.
        """

        # Check if the input-output convex architecture should be used
        if self.configuration['architecture'] == 'input-convex':

            # Build and return the input-output convex neural network (IOCNN)
            return build_iocnn(input_shape, output_shape,
                               self.preprocessed_labels, hyperparameters,
                               squash_output)

        # Build and return the standard neural network (Standard-NN)
        return build_standard_nn(input_shape, output_shape,
                                 self.preprocessed_labels, hyperparameters,
                                 squash_output)

    def compile_and_fit(
            self,
            prediction_model,
            features,
            labels,
            hyperparameters):
        """
        Compile and fit the neural network outcome prediction model to the \
        input data.

        Parameters
        ----------
        prediction_model : object of class `Functional`
            Instance for the provision of the neural network architecture.

        features : tf.float64
            Casted array of input feature values.

        labels : tf.float64
            Casted array of input label values.

        hyperparameters : dict
            Dictionary with the hyperparameter names and values for the \
            neural network outcome prediction model.

        Returns
        -------
        prediction_model : object of class `Functional`
            Instance of the class `Functional`, which holds methods to make \
            predictions from the neural network model.
        """

        # Compile the model
        prediction_model.compile(
            optimizer=optimizer_map[hyperparameters['optimizer']](
                learning_rate=hyperparameters['learning_rate']),
            loss=loss_map[hyperparameters['loss']]())

        # Fit the model with the training data
        prediction_model.fit(
            features,
            labels,
            batch_size=hyperparameters['batch_size'],
            epochs=hyperparameters['epochs'],
            verbose=0,
            callbacks=[
                ReduceLROnPlateau(
                    monitor='loss',
                    min_delta=0,
                    factor=hyperparameters['ReduceLROnPlateau_factor'],
                    patience=hyperparameters['ReduceLROnPlateau_patience'],
                    mode='min',
                    verbose=0),
                EarlyStopping(
                    monitor='loss',
                    min_delta=0,
                    patience=hyperparameters['EarlyStopping_patience'],
                    mode='min',
                    baseline=None,
                    restore_best_weights=True,
                    verbose=0)],
            class_weight={0: (1/sum(labels == 0)*(2*len(labels))),
                          1: (1/sum(labels == 1)*(2*len(labels)))})

        return prediction_model

    def tune_hyperparameters(
            self,
            features,
            labels):
        """
        Tune the hyperparameters of the neural network model via sequential \
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
                train_validate_split = [
                    features[indices[0]], features[indices[1]],
                    labels[indices[0]], labels[indices[1]]
                    ]

                # Fit the preprocessor & transform the training features/labels
                train_validate_split[0], train_validate_split[2] = (
                    preprocessor.fit_transform(
                       train_validate_split[0], train_validate_split[2]))

                # Transform the validation features/labels
                train_validate_split[1], train_validate_split[3] = (
                    preprocessor.transform(
                        train_validate_split[1], train_validate_split[3]))

                # Build the model architecture
                prediction_model = self.build_network(
                    train_validate_split[0].shape[1], labels.ndim,
                    hyperparameters, squash_output=True)

                # Compile and fit the model to the input data
                prediction_model = self.compile_and_fit(
                    prediction_model, train_validate_split[0],
                    train_validate_split[2], hyperparameters)

                # Get the training score
                training_score = -scorers[self.configuration['tune_score']](
                    train_validate_split[2], prediction_model.predict(
                        train_validate_split[0], verbose=0)[:, 0])

                # Get the validation score
                validation_score = -scorers[self.configuration['tune_score']](
                    train_validate_split[3], prediction_model.predict(
                        train_validate_split[1], verbose=0)[:, 0])

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
            hyperparameters = {**proposal['hidden_layers'],
                               'input_neuron_number': proposal[
                                   'input_neuron_number'],
                               'input_activation': proposal[
                                   'input_activation'],
                               'output_activation': 'sigmoid',
                               'input_dropout_rate': proposal[
                                   'input_dropout_rate'],
                               'batch_size': proposal['batch_size'],
                               'epochs': 1000,
                               'learning_rate': proposal['learning_rate'],
                               'optimizer': proposal['optimizer'],
                               'loss': proposal['loss'],
                               'ReduceLROnPlateau_factor': 0.5,
                               'ReduceLROnPlateau_patience': 5,
                               'EarlyStopping_patience': 10}

            # Compute the objective function value (score) across all folds
            fold_scores = map(compute_fold_score, (
                (training_indices, validation_indices)
                for training_indices, validation_indices in (
                    (where(self.configuration['tune_folds'] != number),
                     where(self.configuration['tune_folds'] == number))
                    for number in set(self.configuration['tune_folds']))
                ))

            return {'loss': max(fold_scores),
                    'params': hyperparameters,
                    'status': STATUS_OK}

        # Initialize the data preprocessor
        preprocessor = DataPreprocessor(
            self.preprocessing_steps, verbose=False)

        # Map the score labels to the score functions
        scorers = {'AUC': roc_auc_score, **score_loss_map}

        # Define the search space for the hyperparameters
        space = {
            'hidden_layers': hp.choice(
                'hidden_layers',
                [{'hidden_layer_number': n+1,
                  'hidden_neuron_number': [
                      hp.choice(
                          f'{n+1}L{m+1}_neuron_number',
                          self.configuration[
                              'tune_space']['hidden_neuron_number'])
                      for m in range(n+1)],
                  'hidden_activation': [
                      hp.choice(
                          f'{n+1}L{m+1}_activation',
                          self.configuration[
                              'tune_space']['hidden_activation'])
                      for m in range(n+1)],
                  'hidden_dropout_rate': [
                      hp.choice(
                          f'{n+1}L{m+1}_hidden_dropout',
                          self.configuration[
                              'tune_space']['hidden_dropout_rate'])
                      for m in range(n+1)]}
                 for n in range(self.configuration['max_hidden_layers'])]),
            'input_neuron_number': hp.choice(
                'input_neuron_number',
                self.configuration['tune_space']['input_neuron_number']),
            'input_activation': hp.choice(
                'input_activation',
                self.configuration['tune_space']['input_activation']),
            'input_dropout_rate': hp.choice(
                'input_dropout_rate',
                self.configuration['tune_space']['input_dropout_rate']),
            'batch_size': hp.choice(
                'batch_size',
                self.configuration['tune_space']['batch_size']),
            'learning_rate': hp.uniform(
                'learning_rate',
                self.configuration['tune_space']['learning_rate'][0],
                self.configuration['tune_space']['learning_rate'][1]),
            'optimizer': hp.choice(
                'optimizer',
                self.configuration['tune_space']['optimizer']),
            'loss': hp.choice(
                'loss',
                self.configuration['tune_space']['loss'])
        }

        # Generate a trials object to store the evaluation history
        bayes_trials = Trials()

        # Run the optimization algorithm to get the tuned hyperparameters
        tuned_hyperparameters = fmin(
            fn=partial(objective, trials=bayes_trials, space=space),
            space=space, algo=tpe.suggest, max_evals=self.configuration[
                'tune_evaluations'], trials=bayes_trials, return_argmin=False)

        return tuned_hyperparameters

    def train(
            self,
            features,
            labels):
        """
        Train the neural network outcome prediction model.

        Returns
        -------
        prediction_model : object of class `Functional`
            Instance of the class `Functional`, which holds methods to make \
            predictions from the neural network model.
        """

        # Log a message about the model fitting
        Datahub().logger.display_info("Fitting the model to the data ...")

        # Get the tuned hyperparameters
        tuned_hyperparameters = self.tune_hyperparameters(features, labels)

        # Create a dictionary with the model hyperparameters
        hyperparameters = {**tuned_hyperparameters['hidden_layers'],
                           'input_neuron_number': tuned_hyperparameters[
                               'input_neuron_number'],
                           'input_activation': tuned_hyperparameters[
                               'input_activation'],
                           'output_activation': 'sigmoid',
                           'input_dropout_rate': tuned_hyperparameters[
                               'input_dropout_rate'],
                           'batch_size': tuned_hyperparameters['batch_size'],
                           'epochs': 1000,
                           'learning_rate': tuned_hyperparameters[
                               'learning_rate'],
                           'optimizer': tuned_hyperparameters['optimizer'],
                           'loss': tuned_hyperparameters['loss'],
                           'ReduceLROnPlateau_factor': 0.5,
                           'ReduceLROnPlateau_patience': 5,
                           'EarlyStopping_patience': 10}

        # Build the model architecture
        prediction_model = self.build_network(
            self.preprocessed_features.shape[1], self.preprocessed_labels.ndim,
            hyperparameters, squash_output=True)

        # Compile and fit the model to the preprocessed data
        prediction_model = self.compile_and_fit(
            prediction_model, self.preprocessed_features,
            self.preprocessed_labels, hyperparameters)

        return prediction_model, hyperparameters

    def predict(
            self,
            features,
            squash_output=True):
        """
        Predict the label values from the feature values.

        Parameters
        ----------
        features : ndarray
            Array of input feature values.

        squash_output : bool
            Indicator for the use of a sigmoid activation function in the \
            output layer.

        Returns
        -------
        float or ndarray
            Floating-point label prediction or array of label predictions.
        """

        # Check if the output should be squashed
        if squash_output:

            # Apply the prediction model
            predictor = self.prediction_model

        else:

            # Apply the optimization model
            predictor = self.optimization_model

        # Check if the feature array has only a single row
        if features.shape[0] == 1:

            # Return a single label prediction value
            return predictor(features)[0][0].numpy()

        # Otherwise, return an array with label predictions
        return predictor(features)[:, 0].numpy()

    def predict_oof(
            self,
            features,
            labels):
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
        Datahub().logger.display_info(
            f"Performing {len(set(self.configuration['oof_folds']))}-fold "
            "cross-validation to yield out-of-folds predictions ...")

        def compute_fold_labels(indices):
            """Compute the out-of-folds labels for a single fold."""

            # Get the training and validation splits
            train_validate_split = [
                features[indices[0]], features[indices[1]], labels[indices[0]]]

            # Fit the preprocessor & transform the training features/labels
            train_validate_split[0], train_validate_split[2] = (
                preprocessor.fit_transform(
                    train_validate_split[0], train_validate_split[2]))

            # Transform the validation features
            train_validate_split[1] = preprocessor.transform(
                train_validate_split[1])[0]

            # Build the model architecture
            prediction_model = self.build_network(
                train_validate_split[0].shape[1], labels.ndim,
                self.hyperparameters, squash_output=True)

            # Compile and fit the model to the input data
            prediction_model = self.compile_and_fit(
                prediction_model, train_validate_split[0],
                train_validate_split[2], self.hyperparameters)

            return (indices[1],
                    prediction_model(train_validate_split[1])[:, 0].numpy())

        # Initialize the out-of-folds label prediction array
        oof_prediction = empty((len(labels),))

        # Initialize the data preprocessor
        preprocessor = DataPreprocessor(
            self.preprocessing_steps, verbose=False)

        # Compute the returns (indices and labels) across all folds
        fold_returns = map(compute_fold_labels, (
                (training_indices, validation_indices)
                for training_indices, validation_indices in (
                    (where(self.configuration['tune_folds'] != number),
                     where(self.configuration['tune_folds'] == number))
                    for number in set(self.configuration['tune_folds']))
                ))

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
        """."""

        # Check if the model should be first-time/repeatedly inspected
        if (self.model_label not in Datahub().model_inspections
                or self.updated_model):

            # Compute the model inspections
            self.inspector.compute(
                self.prediction_model, self.hyperparameters, features, labels,
                self.preprocessing_steps, 30, oof_folds)

    def evaluate(
            self,
            features,
            labels):
        """."""

        # Check if the model should be first-time/repeatedly evaluated
        if (self.model_label not in Datahub().model_evaluations
                or self.updated_model):

            # Run the model training and out-of-folds evaluations
            self.evaluator.compute(
                labels, (self.predict(self.preprocessed_features),
                         self.predict_oof(features, labels)))

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
        self.model_path, self.configuration_path, self.hyperparameter_path = (
            f'{base_path}/{filename}' for filename in (
                'model.h5', 'configuration.json', 'hyperparameters.json'))

    def read_model_from_file(self):
        """
        Read the neural network outcome prediction model from the model file \
        path.

        Returns
        -------
        object of class `Functional`
            Instance of the class `Functional`, which holds methods to make \
            predictions from the neural network model.
        """

        # Log a message about the model file reading
        Datahub().logger.display_info(
            f"Reading '{self.model_label}' model from file ...")

        # Open a stream to the model file path
        with File(self.model_path, 'r') as file:

            # Get the network weights from the file
            weights = tuple(file[''.join(('weight', str(i)))][:]
                            for i, _ in enumerate(file))

        # Read the hyperparameters from the file
        hyperparameters = self.read_hyperparameters_from_file()

        # Build the network architecture
        prediction_model = self.build_network(
            self.preprocessed_features.shape[1], self.preprocessed_labels.ndim,
            hyperparameters, squash_output=True)

        # Compile the model
        prediction_model.compile(
            optimizer=optimizer_map[hyperparameters['optimizer']](
                learning_rate=hyperparameters['learning_rate']),
            loss=loss_map[hyperparameters['loss']])

        # Set the network weights for the model
        prediction_model.set_weights(weights)

        return prediction_model

    def write_model_to_file(
            self,
            prediction_model):
        """
        Write the neural network outcome prediction model to the model file \
        path.

        Parameters
        ----------
        prediction_model : object of class `Functional`
            Instance of the class `Functional`, which holds methods to make \
            predictions from the neural network model.
        """

        # Get the network weights of the model
        weights = prediction_model.get_weights()

        # Open a stream to the model file path
        with File(self.model_path, 'w') as file:

            # Add each network weight to the file
            for i, weight in enumerate(weights):
                file.create_dataset(
                    ''.join(('weight', str(i))), data=weight)

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
        Datahub().logger.display_info(
            f"Reading '{self.model_label}' configuration from file ...")

        # Get the configuration
        configuration = jload(
            open(self.configuration_path, 'r', encoding='utf-8'))

        # Loop over the converted keys
        for key in ('feature_values', 'label_values', 'time_variable_values',
                    'tune_folds', 'oof_folds'):

            # Convert the list into an array
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
            Dictionary with information for the modeling, i.e., the dataset, \
            the preprocessing steps, and the hyperparameter search space.
        """

        # Loop over the array keys
        for key in ('feature_values', 'label_values', 'time_variable_values',
                    'tune_folds', 'oof_folds'):

            # Convert the array into a list
            configuration[key] = configuration[key].tolist()

        # Dump the configuration dictionary to the configuration file path
        with open(self.configuration_path, 'w', encoding='utf-8') as file:
            jdump(configuration, file, sort_keys=False, indent=4)

    def read_hyperparameters_from_file(self):
        """
        Read the neural network outcome prediction model hyperparameters from \
        the hyperparameter file path.

        Returns
        -------
        dict
            Dictionary with the hyperparameter names and values for the \
            neural network outcome prediction model.
        """

        # Log a message about the parameter file reading
        Datahub().logger.display_info(
            f"Reading '{self.model_label}' hyperparameters from file ...")

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
            neural network outcome prediction model.
        """

        # Dump the hyperparameter dictionary to the hyperparameter file path
        with open(self.hyperparameter_path, 'w', encoding='utf-8') as file:
            jdump(hyperparameters, file, sort_keys=False, indent=4)
