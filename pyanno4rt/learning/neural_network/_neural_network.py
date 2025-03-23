"""Neural network model."""

# Author: Tim Ortkamp

# %% External package import

from h5py import File
from hyperopt import hp
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import (
    BinaryCrossentropy, BinaryFocalCrossentropy, KLDivergence)
from tensorflow.keras.optimizers import Adam, Ftrl, SGD

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning import MachineLearningModel
from pyanno4rt.learning.neural_network import (
    build_vanilla_iocnn, build_vanilla_nn)

# %% Class definition


class NeuralNetworkModel(MachineLearningModel):
    """
    Neural network model class.

    This class enables building an individual preprocessing pipeline, \
    fitting, making predictions, inspecting, and evaluating the predictive \
    performance of a neural network model.

    See the machine learning model template class \
        :class:`~pyanno4rt.learning_model.frequentist._machine_learning_model.MachineLearningModel`
    for information on the parameters and attributes.

    .. note:: Currently, the hyperparameter search space for the neural \
        network model includes:

            - 'input_neuron_number' : number of neurons for the input layer
            - 'input_activation' : activation function for the input layer
            - 'hidden_neuron_number' : number of neurons for the hidden \
                layer(s)
            - 'hidden_activation' : activation function for the hidden layer(s)
            - 'input_dropout_rate' : dropout rate for the input layer
            - 'hidden_dropout_rate' : dropout rate for the hidden layer(s)
            - 'batch_size' : batch size
            - 'learning_rate' : learning rate
            - 'optimizer' : algorithm for the network optimization
            - 'loss' : loss function for the network optimization

    Attributes
    ----------
    optimization_model : object of class `Functional`
        The object used to represent the optimization model.
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

        # Configure the internal hyperparameter search space
        tune_space = {
            'input_neuron_number': tune_space.get(
                'input_neuron_number', [2**x for x in range(1, 12)]),
            'input_activation': tune_space.get(
                'input_activation', ['elu', 'gelu', 'leaky_relu', 'linear',
                                     'relu', 'softmax', 'softplus', 'swish']),
            'hidden_neuron_number': tune_space.get(
                'hidden_neuron_number', [2**x for x in range(1, 12)]),
            'hidden_activation': tune_space.get(
                'hidden_activation', ['elu', 'gelu', 'leaky_relu', 'linear',
                                      'relu', 'softmax', 'softplus', 'swish']),
            'input_dropout_rate': tune_space.get(
                'input_dropout_rate', [0.0, 0.1, 0.25, 0.5, 0.75]),
            'hidden_dropout_rate': tune_space.get(
                'input_dropout_rate', [0.0, 0.1, 0.25, 0.5, 0.75]),
            'batch_size': tune_space.get('batch_size', [4, 8, 16, 32]),
            'learning_rate': tune_space.get('learning_rate', [1e-5, 1e-2]),
            'optimizer': tune_space.get('optimizer', list(NN_OPTS)),
            'loss': tune_space.get('loss', list(NN_LOSSES))}

        # Configure the hyperopt search space
        hp_space = {
            'hidden_layers': hp.choice(
                'hidden_layers', [
                    {'hidden_layer_number': n+1,
                     'hidden_neuron_number': [
                         hp.choice(
                             f'{n+1}L{m+1}_neuron_number',
                             tune_space['hidden_neuron_number'])
                         for m in range(n+1)],
                     'hidden_activation': [
                         hp.choice(
                             f'{n+1}L{m+1}_activation',
                             tune_space['hidden_activation'])
                         for m in range(n+1)],
                     'hidden_dropout_rate': [
                         hp.choice(
                             f'{n+1}L{m+1}_hidden_dropout',
                             tune_space['hidden_dropout_rate'])
                         for m in range(n+1)]}
                    for n in range(max_hidden_layers)]),
            'input_neuron_number': hp.choice(
                'input_neuron_number', tune_space['input_neuron_number']),
            'input_activation': hp.choice(
                'input_activation', tune_space['input_activation']),
            'input_dropout_rate': hp.choice(
                'input_dropout_rate', tune_space['input_dropout_rate']),
            'batch_size': hp.choice('batch_size', tune_space['batch_size']),
            'learning_rate': hp.uniform(
                'learning_rate', tune_space['learning_rate'][0],
                tune_space['learning_rate'][1]),
            'optimizer': hp.choice('optimizer', tune_space['optimizer']),
            'loss': hp.choice('loss', tune_space['loss'])}

        # Initialize the superclass
        super().__init__(
            model_label, model_folder_path, dataset, preprocessing_steps,
            tune_space, hp_space, tune_evaluations, tune_score,
            inspect_model, evaluate_model, display_options, architecture,
            max_hidden_layers)

        # Get the optimization surrogate of the neural network model
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

        # Check if the proposal has a hidden layer subdictionary
        if 'hidden_layers' in proposal:

            # Get the unpacked hidden layer parameters
            hidden_layers = {**proposal['hidden_layers']}

        else:

            # Get the hidden layer parameters directly
            hidden_layers = {key: proposal[key] for key in (
                'hidden_neuron_number', 'hidden_activation',
                'hidden_dropout_rate')}

        # Build the hyperparameter dictionary
        hyperparameters = {
            **hidden_layers,
            'input_neuron_number': proposal['input_neuron_number'],
            'input_activation': proposal['input_activation'],
            'output_activation': 'sigmoid',
            'input_dropout_rate': proposal['input_dropout_rate'],
            'batch_size': proposal['batch_size'],
            'epochs': 1000,
            'learning_rate': proposal['learning_rate'],
            'optimizer': proposal['optimizer'],
            'loss': proposal['loss'],
            'ReduceLROnPlateau_factor': 0.5,
            'ReduceLROnPlateau_patience': 10,
            'EarlyStopping_patience': 20}

        return hyperparameters

    def get_model_fit(
            self,
            features,
            labels,
            hyperparameters):
        """
        Get the neural network model fit.

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
        prediction_model : object of class `Functional`
            The object used to represent the prediction model.
        """

        # Build the model architecture
        prediction_model = self.build_network(
            features.shape[1], labels.ndim, hyperparameters, True)

        # Compile and fit the model
        prediction_model = self.compile_and_fit(
            prediction_model, features, labels, hyperparameters)

        return prediction_model

    def get_optimization_model(self):
        """
        Get the neural network optimization model.

        Returns
        -------
        object of class `Functional`
            The object used to represent the optimization model.
        """

        # Build the network architecture
        optimization_model = self.build_network(
            self.prediction_model.inputs[0].shape[1],
            self.prediction_model.outputs[0].shape[1],
            self.hyperparameters, False)

        # Compile the model
        optimization_model.compile(
            optimizer=NN_OPTS[self.hyperparameters['optimizer']](
                learning_rate=self.hyperparameters['learning_rate']),
            loss=NN_LOSSES[self.hyperparameters['loss']])

        # Set the network weights
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
            Dictionary with the values of the hyperparameters.

        squash_output : bool
            Indicator for the squashing of the network output.

        Returns
        -------
        object of class 'Functional'
            The object used to represent the prediction model.
        """

        # Check if the input-output convex architecture should be used
        if self.configuration['architecture'] == 'vanilla-input-convex':

            # Get the vanilla input-output convex neural network (IOC-NN)
            return build_vanilla_iocnn(
                input_shape, output_shape, self.configuration['bias'],
                hyperparameters, squash_output)

        # Get the vanilla neural network (NN)
        return build_vanilla_nn(
            input_shape, output_shape, self.configuration['bias'],
            hyperparameters, squash_output)

    def compile_and_fit(
            self,
            prediction_model,
            features,
            labels,
            hyperparameters):
        """
        Compile and fit the neural network model to the input data.

        Parameters
        ----------
        prediction_model : object of class `Functional`
            The shell object used to represent the prediction model.

        features : ndarray
            Values of the input features.

        labels : ndarray
            Values of the input labels.

        hyperparameters : dict
            Dictionary with the values of the hyperparameters.

        Returns
        -------
        prediction_model : object of class `Functional`
            The object used to represent the pre-fitted prediction model.
        """

        # Compile the model
        prediction_model.compile(
            optimizer=NN_OPTS[hyperparameters['optimizer']](
                learning_rate=hyperparameters['learning_rate']),
            loss=NN_LOSSES[hyperparameters['loss']]())

        # Set the callbacks
        callbacks = [
            ReduceLROnPlateau(
                monitor='loss', min_delta=0,
                factor=hyperparameters['ReduceLROnPlateau_factor'],
                patience=hyperparameters['ReduceLROnPlateau_patience'],
                mode='min', verbose=0),
            EarlyStopping(
                monitor='loss', min_delta=0,
                patience=hyperparameters['EarlyStopping_patience'],
                mode='min', baseline=None, restore_best_weights=True,
                verbose=0)]

        # Fit the model with the training data
        prediction_model.fit(
            features, labels, batch_size=hyperparameters['batch_size'],
            epochs=hyperparameters['epochs'], verbose=0, callbacks=callbacks,
            class_weight={0: (1/sum(labels == 0)*(2*len(labels))),
                          1: (1/sum(labels == 1)*(2*len(labels)))})

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

        predictor : object of class `Functional`
            The object used to represent the prediction model.

        Returns
        -------
        float or ndarray
            Value(s) of the predicted label(s).
        """

        # Check if the feature array has only a single row
        if features.shape[0] == 1:

            # Return a single label prediction value
            return predictor(features)[0][0].numpy()

        # Otherwise, return an array with label predictions
        return predictor(features)[:, 0].numpy()

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
        (self.preprocessor_path, self.model_path, self.configuration_path,
         self.hyperparameter_path) = (
             f'{base_path}/{filename}' for filename in (
                 'preprocessor.sav', 'model.h5', 'configuration.json',
                 'hyperparameters.json'))

    def import_model(self):
        """
        Import the neural network model from the model file path.

        Returns
        -------
        object of class `Functional`
            The object used to represent the prediction model.
        """

        # Log a message about the model file reading
        Datahub().logger.display_info(
            f"Reading '{self.model_label}' model from file ...")

        # Open a stream to the model file path
        with File(self.model_path, 'r') as file:

            # Get the network weights from the file
            weights = tuple(file[''.join(('weight', str(i)))][:]
                            for i, _ in enumerate(file))

        # Read the configuration from the file
        configuration = self.import_configuration()

        # Get the number of input features
        input_shape = len(configuration['feature_names'])

        # Check if the label name is a single string
        if isinstance(configuration['label_name'], str):

            # Set the number of output labels to 1
            output_shape = 1

        else:

            # Get the number of output labels
            output_shape = len(configuration['label_name'])

        # Add the label bias to the configuration dictionary
        self.configuration['bias'] = configuration['bias']

        # Read the hyperparameters from the file
        hyperparameters = self.import_hyperparameters(verbose=False)

        # Build the network architecture
        prediction_model = self.build_network(
            input_shape, output_shape, hyperparameters, True)

        # Compile the model
        prediction_model.compile(
            optimizer=NN_OPTS[hyperparameters['optimizer']](
                learning_rate=hyperparameters['learning_rate']),
            loss=NN_LOSSES[hyperparameters['loss']])

        # Set the network weights
        prediction_model.set_weights(weights)

        return prediction_model

    def export_model(self):
        """
        Export the neural network model to a bytes-like object.

        Returns
        -------
        object of class :class:`~h5py._hl.files.File`
            Bytes-like model object.
        """

        # Open a file stream
        with File.in_memory() as file:

            # Loop over the weights
            for i, weight in enumerate(self.prediction_model.get_weights()):

                # Create a dataset
                file.create_dataset(''.join(('weight', str(i))), data=weight)

            return file


# %% Map definitions


NN_LOSSES = {
    'BCE': BinaryCrossentropy,
    'FocalBCE': BinaryFocalCrossentropy,
    'KLD': KLDivergence}

NN_OPTS = {
    'Adam': Adam,
    'Ftrl': Ftrl,
    'SGD': SGD}