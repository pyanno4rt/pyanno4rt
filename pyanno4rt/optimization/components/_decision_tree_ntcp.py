"""Decision tree NTCP component."""

# Author: Tim Ortkamp

# %% External package import

from copy import deepcopy

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model import DataModelHandler
from pyanno4rt.learning_model.frequentist import DecisionTreeModel
from pyanno4rt.optimization.components import MachineLearningComponent
from pyanno4rt.tools import filter_dict

# %% Class definition


class DecisionTreeNTCP(MachineLearningComponent):
    """
    Decision tree NTCP component class.

    This class provides methods to compute the value and the gradient of the \
    decision tree NTCP component, as well as to add the decision tree model.

    Parameters
    ----------
    segment : str
        Name of the segment associated with the component.

    model_parameters : dict
        Dictionary with the data handling & learning model parameters, see \
        the class
        :class:`~pyanno4rt.optimization.components._machine_learning_component_class.MachineLearningComponentClass`.

    component_type : {'constraint', 'objective'}, default='objective'
        Type of the component.

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
    arguments : dict
        Dictionary with the component input arguments (for serialization).

    data_model_handler : object of class \
        :class:`~pyanno4rt.learning_model._data_model_handler.DataModelHandler`
        The object used to handle the dataset, the feature map generation and \
        the feature (re-)calculation.

    model : object of class \
        :class:`~pyanno4rt.learning_model.frequentist._decision_tree.DecisionTreeModel`
        The object used to preprocess, tune, train, inspect and evaluate the \
        decision tree model.

    parameter_value : list
        Decision tree model parameters.

    bounds : list
        See 'Parameters'.
    """

    def __init__(
            self,
            segment,
            model_parameters,
            component_type='objective',
            embedding='active',
            weight=1.0,
            rank=1,
            bounds=None,
            link=None,
            identifier=None,
            display=True):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(
            name='Decision Tree NTCP',
            segment=segment,
            component_type=component_type,
            parameter_name=(),
            parameter_category=(),
            model_parameters=model_parameters,
            embedding=embedding,
            weight=weight,
            rank=rank,
            bounds=bounds,
            link=link,
            identifier=identifier,
            display=display)

        # Set the input arguments
        self.arguments = filter_dict(
            locals(), remove_keys=('self', '__class__'))

    def to_dict(self):
        """Return the component input dictionary."""

        # Get the parameter dictionary
        dictionary = deepcopy(self.arguments)

        # Serialize the data columns
        dictionary['model_parameters']['data_columns'] = [
            item.to_dict()
            for item in dictionary['model_parameters']['data_columns']]

        return {'Decision Tree NTCP': dictionary}

    def add_model(self):
        """Add the decision tree model to the component."""

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the model addition
        hub.logger.display_info(
            f"Adding decision tree model for '{self.name}' ...")

        # Initialize the data model handler
        self.data_model_handler = DataModelHandler(
            model_label=self.model_parameters['model_label'],
            model_folder_path=self.model_parameters['model_folder_path'],
            data_path=self.model_parameters['data_path'],
            data_columns=self.model_parameters['data_columns'],
            tune_splits=self.model_parameters['tune_splits'],
            tune_repeats=self.model_parameters['tune_repeats'],
            oof_splits=self.model_parameters['oof_splits'],
            oof_repeats=self.model_parameters['oof_repeats'],
            write_features=self.model_parameters['write_features'])

        # Integrate the model-related classes
        self.data_model_handler.integrate()

        # Initialize the decision tree model
        self.model = DecisionTreeModel(
            model_label=self.model_parameters['model_label'],
            model_folder_path=self.model_parameters['model_folder_path'],
            dataset=hub.datasets[self.model_parameters['model_label']],
            preprocessing_steps=self.model_parameters['preprocessing_steps'],
            tune_space=self.model_parameters['tune_space'],
            tune_evaluations=self.model_parameters['tune_evaluations'],
            tune_score=self.model_parameters['tune_score'],
            inspect_model=self.model_parameters['inspect_model'],
            evaluate_model=self.model_parameters['evaluate_model'],
            display_options=self.model_parameters['display_options'])

        # Get the decision tree model parameters
        self.parameter_value = []

        # Transform the component bounds
        self.bounds = sorted(-bound for bound in self.bounds)

    def compute_value(
            self,
            dose,
            segment):
        """
        Compute the function value.

        Parameters
        ----------
        dose : tuple
            Tuple with the dose values.

        segment : tuple
            Tuple with the segment names.

        Returns
        -------
        float
            Function value.
        """

        # Compute the feature vector
        raw_features = self.data_model_handler.feature_calculator.featurize(
            dose, segment)

        # Preprocess the feature vector
        preprocessed_features = self.model.preprocess(raw_features)

        return self.model.predict(
            preprocessed_features, self.model.optimization_model)

    def compute_gradient(
            self,
            dose,
            segment):
        """
        Compute the gradient vector.

        Parameters
        ----------
        dose : tuple
            Tuple with the dose values.

        segment : tuple
            Tuple with the segment names.

        Returns
        -------
        ndarray
            Gradient vector.
        """

        # Get the feature calculator
        feature_calculator = self.data_model_handler.feature_calculator

        # Compute the feature vector
        raw_features = feature_calculator.featurize(dose, segment)

        # Preprocess the feature vector
        preprocessed_features = self.model.preprocess(raw_features)

        # Compute the model gradient
        model_gradient = self.model.optimization_model.gradientize(
            preprocessed_features)

        # Compute the preprocessing pipeline gradient
        preprocessing_gradient = (
            self.model.preprocessor.gradientize(raw_features))

        # Compute the feature gradient
        feature_gradient = feature_calculator.gradientize(dose, segment)

        return (model_gradient * preprocessing_gradient) @ feature_gradient
