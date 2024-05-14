"""Neural network NTCP component."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import array
from tensorflow import cast, float64, GradientTape

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model import DataModelHandler
from pyanno4rt.learning_model.frequentist import NeuralNetworkModel
from pyanno4rt.optimization.components import MachineLearningComponentClass
from pyanno4rt.tools import inverse_sigmoid

# %% Class definition


class NeuralNetworkNTCP(MachineLearningComponentClass):
    """
    Neural network NTCP component class.

    This class provides methods to compute the value and the gradient of the \
    neural network NTCP component, as well as to add the neural network model.

    Parameters
    ----------
    model_parameters : dict
        Dictionary with the data handling & learning model parameters, see \
        the class
        :class:`~pyanno4rt.optimization.components._machine_learning_component_class.MachineLearningComponentClass`.

    embedding : {'active', 'passive'}, default='active'
        Mode of embedding for the component. In 'passive' mode, the component \
        value is computed and tracked, but not considered in the optimization \
        problem, unlike in 'active' mode.

    weight : int or float, default=1.0
        Weight of the component function.

    rank : int, default=1
        Rank of the component in the lexicographic order.

    bounds : None or list, default=None
        Constraint bounds for the component.

    link : None or list, default=None
        Other segments used for joint evaluation.

    identifier : None or str, default=None
        Additional string for naming the component.

    display : bool, default=True
        Indicator for the display of the component.

    Attributes
    ----------
    data_model_handler : object of class \
        :class:`~pyanno4rt.learning_model._data_model_handler.DataModelHandler`
        The object used to handle the dataset, the feature map generation and \
        the feature (re-)calculation.

    model : object of class \
        :class:`~pyanno4rt.learning_model.frequentist._neural_network.NeuralNetworkModel`
        The object used to preprocess, tune, train, inspect and evaluate the \
        neural network model.

    parameter_value : list
        Value of the neural network model parameters.

    bounds : list
        See 'Parameters'. Transformed by the inverse sigmoid function.
    """

    def __init__(
            self,
            model_parameters,
            embedding='active',
            weight=1.0,
            rank=1,
            bounds=None,
            link=None,
            identifier=None,
            display=True):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(name='Neural Network NTCP',
                         parameter_name=('(weight, bias)',),
                         parameter_category=('parameter',),
                         model_parameters=model_parameters,
                         embedding=embedding,
                         weight=weight,
                         rank=rank,
                         bounds=bounds,
                         link=link,
                         identifier=identifier,
                         display=display)

    def add_model(self):
        """Add the neural network model to the component."""

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the model addition
        hub.logger.display_info(
            f"Adding {self.model_parameters['architecture']} neural network "
            f"model for '{self.name}' ...")

        # Initialize the data model handler
        self.data_model_handler = DataModelHandler(
            model_label=self.model_parameters['model_label'],
            data_path=self.model_parameters['data_path'],
            feature_filter=self.model_parameters['feature_filter'],
            label_name=self.model_parameters['label_name'],
            label_bounds=self.model_parameters['label_bounds'],
            time_variable_name=self.model_parameters['time_variable_name'],
            label_viewpoint=self.model_parameters['label_viewpoint'],
            tune_splits=self.model_parameters['tune_splits'],
            oof_splits=self.model_parameters['oof_splits'],
            fuzzy_matching=self.model_parameters['fuzzy_matching'],
            write_features=self.model_parameters['write_features'])

        # Integrate the model-related classes
        self.data_model_handler.integrate()

        # Initialize the neural network model
        self.model = NeuralNetworkModel(
            model_label=self.model_parameters['model_label'],
            model_folder_path=self.model_parameters['model_folder_path'],
            dataset=hub.datasets[self.model_parameters['model_label']],
            preprocessing_steps=self.model_parameters['preprocessing_steps'],
            architecture=self.model_parameters['architecture'],
            max_hidden_layers=self.model_parameters['max_hidden_layers'],
            tune_space=self.model_parameters['tune_space'],
            tune_evaluations=self.model_parameters['tune_evaluations'],
            tune_score=self.model_parameters['tune_score'],
            inspect_model=self.model_parameters['inspect_model'],
            evaluate_model=self.model_parameters['evaluate_model'],
            display_options=self.model_parameters['display_options'])

        # Get the neural network model parameters
        self.parameter_value = list(
            weight for weights in (
                layer.flatten().astype(float) if len(layer.shape) == 2
                else layer.astype(float)
                for layer in self.model.prediction_model.get_weights())
            for weight in weights)

        # Transform the component bounds
        self.bounds = sorted(
            self.weight*inverse_sigmoid(bound) for bound in self.bounds)

    def compute_value(
            self,
            *args):
        """
        Compute the component value.

        Parameters
        ----------
        *args : tuple
            Keyworded parameters, where args[0] must be the dose vector(s) to \
            evaluate and args[1] the corresponding segment(s).

        Returns
        -------
        float
            Value of the component function.
        """

        # Compute the feature vector from the dose vector(s) and segment(s)
        raw_features = self.data_model_handler.feature_calculator.featurize(
            args[0], args[1])

        # Preprocess and cast the feature vector
        preprocessed_features = cast(
            self.model.preprocess(raw_features), float64)

        return self.model.predict(preprocessed_features, squash_output=False)

    def compute_gradient(
            self,
            *args):
        """
        Compute the component gradient.

        Parameters
        ----------
        *args : tuple
            Keyworded parameters, where args[0] must be the dose vector(s) to \
            evaluate and args[1] the corresponding segment(s).

        Returns
        -------
        ndarray
            Value of the component gradient.
        """

        # Get the feature calculator
        feature_calculator = self.data_model_handler.feature_calculator

        # Compute the feature vector from the dose vector(s) and segment(s)
        raw_features = feature_calculator.featurize(args[0], args[1])

        # Preprocess and cast the feature vector
        preprocessed_features = cast(
            self.model.preprocess(raw_features), float64)

        # Open a gradient tape stream
        with GradientTape() as tape:

            # Watch the gradient operations on the preprocessed features
            tape.watch(preprocessed_features)

            # Compute the model output from the features
            output = self.model.optimization_model(preprocessed_features)

            # Compute the model gradient
            model_gradient = array(
                tape.gradient(output, preprocessed_features)).reshape(-1)

        # Compute the preprocessing pipeline gradient
        preprocessing_gradient = (
            self.model.preprocessor.gradientize(raw_features))

        # Compute the feature gradient
        feature_gradient = feature_calculator.gradientize(args[0], args[1])

        return (model_gradient * preprocessing_gradient) @ feature_gradient
