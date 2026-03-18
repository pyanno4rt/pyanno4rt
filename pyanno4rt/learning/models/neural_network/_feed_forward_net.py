"""Feed-forward neural network model."""

# Author: Tim Ortkamp

# %% External package import

from warnings import filterwarnings

from numpy import array
from tensorflow import cast, clip_by_value, float64, GradientTape, transpose
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

# %% Internal package import

from pyanno4rt.learning.models import MachineLearningModel
from pyanno4rt.learning._maps import NETWORK_LOSSES, NETWORK_OPTIMIZERS
from pyanno4rt.learning.models.neural_network import build_fnn, build_icnn
from pyanno4rt.validation import validate_item_in_set, validate_type

# %% Set package options

filterwarnings(action='ignore')

# %% Class definition


class FeedForwardNet(MachineLearningModel):
    """
    Feed-forward neural network model class.

    This class implements methods to handle feed-forward neural network models.

    Parameters
    ----------
    label : str
        Label for the learning model.

    architecture : {'FNN', 'ICNN'}
        Network architecture.

        Currently available:

        - 'FNN': "vanilla" neural network
        - 'ICNN': input-convex neural network

    dataset : object of class \
        :class:`~pyanno4rt.learning.datasets._tabular_dataset.TabularDataset`
        The object used to represent the dataset.

    preprocessor : None or object of class \
        :class:`~pyanno4rt.learning.preprocessing._tabular_preprocessor.TabularPreprocessor`,\
        default=None
        The object used to represent the data preprocessor.

    tuner : None or object of class \
        :class:`~pyanno4rt.learning.tuning._bayes_hp_tuner.BayesHPTuner`\
        :class:`~pyanno4rt.learning.tuning._grid_hp_tuner.GridHPTuner`\
        :class:`~pyanno4rt.learning.tuning._randomized_hp_tuner.RandomizedHPTuner`,\
        default=None
        The object used to represent the hyperparameter tuner.

    inspector : None or object of class \
        :class:`~pyanno4rt.learning.inspection._model_inspector.ModelInspector`,\
        default=None
        The object used to represent the model inspector.

    evaluator : None or object of class \
        :class:`~pyanno4rt.learning.evaluation._model_evaluator.ModelEvaluator`,\
        default=None
        The object used to represent the model evaluator.

    model_path : None or str, default=None
        Path to an external model.

    Attributes
    ----------
    architecture : {'FNN', 'ICNN'}
        See 'Parameters'.

    hyperparameters : dict
        Dictionary with the model hyperparameters.

    predictor : None or object of class :class:`~tensorflow.keras.Model`
        The object used to represent the prediction model.

    Notes
    -----
    See :class:`~pyanno4rt.learning.models._machine_learning_model.MachineLearningModel`\
    for details on the inherited attributes.
    """

    def __init__(
            self,
            label,
            architecture,
            dataset,
            preprocessor=None,
            tuner=None,
            inspector=None,
            evaluator=None,
            model_path=None):

        # Call the superclass constructor
        super().__init__(
            label=label,
            dataset=dataset,
            preprocessor=preprocessor,
            tuner=tuner,
            inspector=inspector,
            evaluator=evaluator,
            model_path=model_path)

        # Get the architecture
        self.architecture = architecture

        # Validate the architecture
        validate_type('architecture', architecture, str)
        validate_item_in_set('architecture', architecture, ('FNN', 'ICNN'))

        # Extend the input arguments
        self.arguments |= {'architecture': architecture}

        # Initialize the hyperparameters
        self.hyperparameters = {
            'hidden_layer_number': 1,
            'hidden_neuron_number': [32],
            'hidden_activation': ['relu'],
            'hidden_dropout_rate': [0.0],
            'output_activation': 'sigmoid',
            'batch_size': 16,
            'epochs': 100,
            'learning_rate': 1e-3,
            'optimizer': 'adam',
            'loss': 'binary_crossentropy',
            'ReduceLROnPlateau_factor': 0.1,
            'ReduceLROnPlateau_patience': 5,
            'EarlyStopping_patience': 10}

        # Initialize the predictor
        self.predictor = None

    def update_hyperparameters(
            self,
            proposal):
        """
        Update the hyperparameters from a search proposal.

        Parameters
        ----------
        proposal : dict
            Proposal for the tunable hyperparameters.
        """

        # Get the hidden layer subdictionary, if applicable
        hidden_layers = proposal.pop('hidden_layers', {})

        # Update the proposal
        proposal |= hidden_layers

        # Update the hyperparameters
        self.hyperparameters |= {
            key: proposal[key]
            for key in proposal.keys() & self.hyperparameters.keys()}

    def fit_predictor(
            self,
            features,
            labels):
        """
        Fit the model.

        Parameters
        ----------
        features : ndarray
            Feature values.

        labels : ndarray
            Label values.
        """

        # Set the callbacks
        callbacks = [
            ReduceLROnPlateau(
                monitor='loss', min_delta=0,
                factor=self.hyperparameters['ReduceLROnPlateau_factor'],
                patience=self.hyperparameters['ReduceLROnPlateau_patience'],
                mode='min', verbose=0),
            EarlyStopping(
                monitor='loss', min_delta=0,
                patience=self.hyperparameters['EarlyStopping_patience'],
                mode='min', baseline=None, restore_best_weights=True,
                verbose=0)]

        # Make the predictor
        self._make_predictor(features.shape[1])

        # Fit the predictor
        self.predictor.fit(
            features, labels, batch_size=self.hyperparameters['batch_size'],
            epochs=self.hyperparameters['epochs'], verbose=0,
            callbacks=callbacks, class_weight={
                0: (1/sum(labels == 0)*(2*len(labels))),
                1: (1/sum(labels == 1)*(2*len(labels)))})

    def _make_predictor(
            self,
            number_of_inputs):
        """
        Make the predictor.

        Parameters
        ----------
        number_of_inputs : int
            Number of input channels.
        """

        # Check if the "vanilla" architecture should be used
        if self.architecture == 'FNN':

            # Build the FNN
            self.predictor = build_fnn(
                number_of_inputs, 1, self.hyperparameters)

        # Else, check if the input-convex architecture should be used
        elif self.architecture == 'ICNN':

            # Build the ICNN
            self.predictor = build_icnn(
                number_of_inputs, 1, self.hyperparameters)

        # Compile the predictor
        self.predictor.compile(
            optimizer=NETWORK_OPTIMIZERS[self.hyperparameters['optimizer']](
                learning_rate=float(self.hyperparameters['learning_rate'])),
            loss=NETWORK_LOSSES[self.hyperparameters['loss']]())

    def predict(
            self,
            features):
        """
        Predict the label value(s).

        Parameters
        ----------
        features : ndarray
            Feature values.

        Returns
        -------
        float or ndarray
            Predicted label value(s).
        """

        # Cast the features
        features = cast(features, float64)

        # Check if the feature array has only a single row
        if features.shape[0] == 1:

            # Return a single label prediction value
            return self.predictor(features)[0][0].numpy().astype(float)

        # Else, return an array with label predictions
        return self.predictor(features)[:, 0].numpy().astype(float)

    def _predictor_gradient(
            self,
            preprocessed_features):
        """
        Calculate the predictor gradient.

        Parameters
        ----------
        preprocessed_features : ndarray
            Preprocessed feature values.

        Returns
        -------
        ndarray
            Predictor gradient w.r.t the preprocessed features.
        """

        # Cast the preprocessed features
        preprocessed_features = cast(preprocessed_features, float64)

        # Record operations for auto-differentiation
        with GradientTape() as tape:

            # Trace the preprocessed features
            tape.watch(preprocessed_features)

            # Calculate the model prediction
            prediction = self.predictor(preprocessed_features)

            # Clip the prediction for numerical stability
            prediction = clip_by_value(prediction, 1e-6, 1-1e-6)

            # Get the tape gradient
            tape_gradient = (prediction-prediction**2)*array(tape.gradient(
                prediction, preprocessed_features)).reshape(-1)

            return transpose(tape_gradient).numpy().flatten().astype(float)

    def _load_predictor(self):
        """Load the predictor."""

        # Load the predictor
        self.predictor = load_model(self.model_path+'/predictor.keras')

    def _load_hyperparameters(self):
        """Load the hyperparameters from the predictor."""

        # Get the dense layers
        dense_layers = [
            layer for layer in self.predictor.layers[:-1]
            if hasattr(layer, 'units') or hasattr(layer, 'filters')]

        # Get the hyperparameters
        self.hyperparameters = {
            'hidden_layer_number': len(dense_layers),
            'hidden_neuron_number': [layer.units for layer in dense_layers],
            'hidden_activation': [
                layer.activation.__name__ for layer in dense_layers],
            'hidden_dropout_rate': [
                layer.rate for layer in self.predictor.layers[:-1]
                if hasattr(layer, 'rate')],
            'output_activation': 'sigmoid',
            'batch_size': 16,
            'epochs': 100,
            'learning_rate': self.predictor.optimizer.learning_rate.numpy(),
            'optimizer': self.predictor.optimizer.name.lower(),
            'loss': self.predictor.loss.name,
            'ReduceLROnPlateau_factor': 0.1,
            'ReduceLROnPlateau_patience': 5,
            'EarlyStopping_patience': 10}

    def _save_predictor(
            self,
            path):
        """
        Save the predictor.

        Parameters
        ----------
        path : str
            Path for storing the predictor.
        """

        # Save the predictor
        self.predictor.save(path+'/predictor.keras')
