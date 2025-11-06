"""Random forest outcome component."""

# Author: Tim Ortkamp

# %% External package import

from copy import deepcopy

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning import DataModelHandler, ModelParameters
from pyanno4rt.learning.forest import RandomForestModel
from pyanno4rt.optimization.components import MachineLearningComponent
from pyanno4rt.tools import filter_dict

# %% Class definition


class RandomForestOutcome(MachineLearningComponent):
    """
    Random forest outcome component class.

    This class provides methods to compute the value and the gradient of the \
    random forest outcome component, as well as to add the random forest model.

    Parameters
    ----------
    segment : str
        Name of the segment associated with the component.

    outcome_type : {'NTCP', 'TCP'}
        Type of the outcome variable.

    model_parameters : object of class \
        :class:`~pyanno4rt.learning._model_parameters.ModelParameters`
        The object used to represent the learning model parameters.

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

    transform : bool, default=False
        Indicator for the transformation of the outcome function.

    identifier : None or str, default=None
        Additional string for naming the component.

    display : bool, default=True
        Indicator for the display of the component.

    Attributes
    ----------
    arguments : dict
        Dictionary with the component input arguments (for serialization).

    data_model_handler : object of class \
        :class:`~pyanno4rt.learning._data_model_handler.DataModelHandler`
        The object used to handle the dataset, the feature map generation and \
        the feature (re-)calculation.

    model : object of class \
        :class:`~pyanno4rt.learning.forest._random_forest.RandomForestModel`
        The object used to preprocess, tune, train, inspect and evaluate the \
        random forest model.

    parameter_value : list
        Random forest model parameters.
    """

    def __init__(
            self,
            segment,
            outcome_type,
            model_parameters,
            component_type='objective',
            embedding='active',
            weight=1.0,
            rank=1,
            bounds=None,
            link=None,
            transform=False,
            identifier=None,
            display=True):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(
            name='Random Forest Outcome',
            segment=segment,
            outcome_type=outcome_type,
            component_type=component_type,
            parameter_name=(),
            parameter_category=(),
            model_parameters=model_parameters,
            embedding=embedding,
            weight=weight,
            rank=rank,
            bounds=bounds,
            link=link,
            transform=transform,
            identifier=identifier,
            display=display)

        # Set the input arguments
        self.arguments = filter_dict(
            locals(), remove_keys=('self', '__class__'))

    def to_dict(self):
        """Serialize the component into a dictionary."""

        # Get the parameter dictionary
        dictionary = deepcopy(self.arguments)

        # Serialize the model parameters
        dictionary['model_parameters'] = (
            dictionary['model_parameters'].to_dict())

        return {self.name: dictionary}

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the component from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the component parameters.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.optimization.components._random_forest_outcome.RandomForestOutcome`
            The object used to handle the component parameters.
        """

        # Deserialize the model parameters
        dictionary['model_parameters'] = ModelParameters.from_dict(
            dictionary['model_parameters'])

        return cls(**dictionary)

    def add_model(self):
        """Add the random forest model to the component."""

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the model addition
        hub.logger.display_info(
            f"Adding random forest model for '{self.name}' ...")

        # Initialize the data model handler
        self.data_model_handler = DataModelHandler(
            model_label=self.model_parameters.model_label,
            model_folder_path=self.model_parameters.model_folder_path,
            data_path=self.model_parameters.data_path,
            data_columns=self.model_parameters.data_columns,
            tune_splits=self.model_parameters.tune_splits,
            tune_repeats=self.model_parameters.tune_repeats,
            oof_splits=self.model_parameters.oof_splits,
            oof_repeats=self.model_parameters.oof_repeats,
            write_features=self.model_parameters.write_features)

        # Integrate the model-related classes
        self.data_model_handler.integrate()

        # Initialize the random forest model
        self.model = RandomForestModel(
            model_label=self.model_parameters.model_label,
            model_folder_path=self.model_parameters.model_folder_path,
            dataset=hub.datasets[self.model_parameters.model_label],
            preprocessing_steps=self.model_parameters.preprocessing,
            tune_space=self.model_parameters.tune_space,
            tune_evaluations=self.model_parameters.tune_evaluations,
            tune_score=self.model_parameters.tune_score,
            inspect_model=self.model_parameters.inspect,
            evaluate_model=self.model_parameters.evaluate,
            display_options=self.model_parameters.display_options)

        # Get the random forest model parameters
        self.parameter_value = []

        # Check if the outcome is 'TCP'
        if self.outcome_type == 'TCP':

            # Convert the bounds
            self.bounds = sorted(
                self.weight*self.reverse(bound) for bound in self.bounds)

        else:

            # Convert the bounds
            self.bounds = [
                self.weight*self.reverse(bound) for bound in self.bounds]

    def translate(
            self,
            value):
        """
        Translate function values to outcome values.

        Parameters
        ----------
        value : int, float, tuple or list
            Function value to translate.

        Returns
        -------
        int, float, tuple or list
            Outcome value.
        """

        # Get the sign
        sign = (-1)**(self.outcome_type == 'TCP')

        # Check if the value is an iterable
        if isinstance(value, (tuple, list)):

            # Return a list of outcome values
            return [sign*val for val in value]

        # Return a single outcome value
        return sign*value

    def reverse(
            self,
            value):
        """
        Reverse outcome values to function values.

        Parameters
        ----------
        value : int, float, tuple or list
            Outcome value to reverse.

        Returns
        -------
        int, float, tuple or list
            Function value.
        """

        # Get the sign
        sign = (-1)**(self.outcome_type == 'TCP')

        # Check if the value is an iterable
        if isinstance(value, (tuple, list)):

            # Return a list of function values
            return [sign*val for val in value]

        # Return a single function value
        return sign*value

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

        # Get the outcome prediction
        prediction = self.model.predict(
            preprocessed_features, self.model.optimization_model)

        # Clip the prediction for numerical stability
        prediction = max(1e-6, min(prediction, 1-1e-6))

        return self.reverse(prediction)

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
        model_gradient = (
            (-1)**(self.outcome_type == 'TCP')
            *self.model.optimization_model.gradientize(preprocessed_features))

        # Compute the preprocessing pipeline gradient
        preprocessing_gradient = (
            self.model.preprocessor.gradientize(raw_features))

        # Compute the feature gradient
        feature_gradient = feature_calculator.gradientize(dose, segment)

        return (model_gradient * preprocessing_gradient) @ feature_gradient
