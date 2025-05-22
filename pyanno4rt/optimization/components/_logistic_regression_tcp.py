"""Logistic regression TCP component."""

# Author: Tim Ortkamp

# %% External package import

from copy import deepcopy
from numpy import array, dot

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning import DataModelHandler, ModelParameters
from pyanno4rt.learning.logistic import LogisticRegressionModel
from pyanno4rt.optimization.components import MachineLearningComponent
from pyanno4rt.tools import filter_dict, inverse_sigmoid, sigmoid

# %% Class definition


class LogisticRegressionTCP(MachineLearningComponent):
    """
    Logistic regression TCP component class.

    This class provides methods to compute the value and the gradient of the \
    logistic regression TCP component, as well as to add the logistic \
    regression model.

    Parameters
    ----------
    segment : str
        Name of the segment associated with the component.

    model_parameters : object of class \
        :class:`~pyanno4rt.learning._model_parameters.ModelParameters`
        The object used to represent the learning model parameters.

    component_type : {'objective', 'constraint'}, default='objective'
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
        :class:`~pyanno4rt.learning._data_model_handler.DataModelHandler`
        The object used to handle the dataset, the feature map generation and \
        the feature (re-)calculation.

    model : object of class \
        :class:`~pyanno4rt.learning.logistic._logistic_regression.LogisticRegressionModel`
        The object used to preprocess, tune, train, inspect and evaluate the \
        logistic regression model.

    parameter_value : list
        Logistic regression model coefficients.

    intercept_value : None or list
        Logistic regression model intercept.

    bounds : list
        See 'Parameters'. Transformed by the inverse sigmoid function.
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
            name='Logistic Regression TCP',
            segment=segment,
            component_type=component_type,
            parameter_name=('beta',),
            parameter_category=('coefficient',),
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

        # Initialize the intercept value
        self.intercept_value = None

    def to_dict(self):
        """Serialize the component into a dictionary."""

        # Get the parameter dictionary
        dictionary = deepcopy(self.arguments)

        # Serialize the data columns
        dictionary['model_parameters'] = (
            dictionary['model_parameters'].to_dict())

        return {'Logistic Regression TCP': dictionary}

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
            :class:`~pyanno4rt.optimization.components._logistic_regression_tcp.LogisticRegressionTCP`
            The object used to handle the component parameters.
        """

        # Deserialize the data columns
        dictionary['model_parameters'] = ModelParameters.from_dict(
            dictionary['model_parameters'])

        return cls(**dictionary)

    def add_model(self):
        """Add the logistic regression model to the component."""

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the model addition
        hub.logger.display_info(
            f"Adding logistic regression model for '{self.name}' ...")

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

        # Initialize the logistic regression model
        self.model = LogisticRegressionModel(
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

        # Get the logistic regression model parameters
        self.parameter_value = list(self.model.prediction_model.coef_[0])
        self.intercept_value = list(self.model.prediction_model.intercept_)

        # Transform the component bounds
        self.bounds = sorted(-inverse_sigmoid(bound) for bound in self.bounds)

    def reverse(
            self,
            value):
        """
        Reverse the component value(s) to the outcome value(s).

        Parameters
        ----------
        value : int, float, tuple or list
            Component value(s).

        Returns
        -------
        float or tuple
            Outcome value(s).
        """

        # Check if the passed value is tuple or a list
        if isinstance(value, (tuple, list)):

            return tuple(1-val for val in sigmoid(value, 1, 0))

        return 1-sigmoid(value, 1, 0)

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

        return -(
            (dot(preprocessed_features, self.parameter_value)
             + self.intercept_value)[0])

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

        # Get the model gradient
        model_gradient = -array(self.parameter_value)

        # Compute the preprocessing pipeline gradient
        preprocessing_gradient = (
            self.model.preprocessor.gradientize(raw_features))

        # Compute the feature gradient
        feature_gradient = feature_calculator.gradientize(dose, segment)

        return (model_gradient * preprocessing_gradient) @ feature_gradient
