"""Logistic regression outcome component."""

# Author: Tim Ortkamp

# %% External package import

from copy import deepcopy
from numpy import array

# %% Internal package import

from pyanno4rt.learning.models.logistic import LogisticRegression
from pyanno4rt.logging import get_logger
from pyanno4rt.optimization.components import MachineLearningComponent
from pyanno4rt.tools import (
    filter_dict, inverse_salu, inverse_sigmoid, salu, sigmoid)

# %% Class definition


class LogisticRegressionOutcome(MachineLearningComponent):
    """
    Logistic regression outcome component class.

    This class provides methods to compute the value and the gradient of the \
    logistic regression outcome component, as well as to add the logistic \
    regression model.

    Parameters
    ----------
    segment : str
        Name of the segment associated with the component.

    outcome_type : {'NTCP', 'TCP'}
        Type of the outcome variable.

    model : object of class \
        :class:`~pyanno4rt.learning.models.logistic._logistic_regression.LogisticRegression`
        The object used to represent the logistic regression outcome model.

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

    model : object of class \
        :class:`~pyanno4rt.learning.models.logistic._logistic_regression.LogisticRegression`
        The object used to represent the logistic regression outcome model.

    parameter_value : list
        Logistic regression model coefficients.
    """

    def __init__(
            self,
            segment,
            outcome_type,
            model,
            component_type='objective',
            embedding='active',
            weight=1.0,
            rank=1,
            bounds=None,
            transform=False,
            identifier=None):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(
            name='Logistic Regression Outcome',
            segment=segment,
            outcome_type=outcome_type,
            component_type=component_type,
            parameter_name=('beta',),
            parameter_category=('coefficient',),
            model=model,
            embedding=embedding,
            weight=weight,
            rank=rank,
            bounds=bounds,
            transform=transform,
            identifier=identifier)

        # Set the input arguments
        self.arguments = filter_dict(
            locals(), remove_keys=('self', '__class__'))

        # Convert the bounds
        self.bounds = sorted(self.reverse(bound) for bound in self.bounds)

    def to_dict(self):
        """Serialize the component into a dictionary."""

        # Get the parameter dictionary
        dictionary = deepcopy(self.arguments)

        # Serialize the model
        dictionary['model'] = (
            dictionary['model'].to_dict())

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
            :class:`~pyanno4rt.optimization.components._logistic_regression_outcome.LogisticRegressionOutcome`
            The object used to handle the component parameters.
        """

        # Deserialize the model
        dictionary['model'] = LogisticRegression.from_dict(dictionary['model'])

        return cls(**dictionary)

    def add_model(self):
        """Add the logistic regression model to the component."""

        # Log a message about the model addition
        get_logger().info(
            "Adding logistic regression model for '%s' ...", self.name)

        # # Initialize the data model handler
        # self.data_model_handler = DataModelHandler(
        #     model_label=self.model_parameters.model_label,
        #     model_folder_path=self.model_parameters.model_folder_path,
        #     data_path=self.model_parameters.data_path,
        #     data_columns=self.model_parameters.data_columns,
        #     tune_splits=self.model_parameters.tune_splits,
        #     tune_repeats=self.model_parameters.tune_repeats,
        #     oof_splits=self.model_parameters.oof_splits,
        #     oof_repeats=self.model_parameters.oof_repeats,
        #     write_features=self.model_parameters.write_features)

        # # Integrate the model-related classes
        # self.data_model_handler.integrate()

        # # Initialize the logistic regression model
        # self.model = LogisticRegressionModel(
        #     model_label=self.model_parameters.model_label,
        #     model_folder_path=self.model_parameters.model_folder_path,
        #     dataset=hub.datasets[self.model_parameters.model_label],
        #     preprocessing_steps=self.model_parameters.preprocessing,
        #     tune_space=self.model_parameters.tune_space,
        #     tune_evaluations=self.model_parameters.tune_evaluations,
        #     tune_score=self.model_parameters.tune_score,
        #     inspect_model=self.model_parameters.inspect,
        #     evaluate_model=self.model_parameters.evaluate,
        #     display_options=self.model_parameters.display_options)

        # Get the logistic regression model parameters
        self.parameter_value = list(self.model.prediction_model.coef_[0])

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

        # Check if the transformation should be applied
        if self.transform:

            # Return the transformed outcome value
            return sigmoid(inverse_salu(value, sign))

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

        # Check if the transformation should be applied
        if self.transform:

            # Return the transformed function value
            return salu(inverse_sigmoid(value), sign)

        # Check if the value is an iterable
        if isinstance(value, (tuple, list)):

            # Return a list of function values
            return [sign*val for val in value]

        # Return a single function value
        return sign*value

    def compute_value(
            self,
            dose):
        """
        Compute the function value.

        Parameters
        ----------
        dose : tuple
            Tuple with the dose arrays.

        Returns
        -------
        float
            Function value.
        """

        # Compute the feature vector
        raw_features = self.model.featurize(dose, self.segment)

        # Preprocess the feature vector
        preprocessed_features = self.model.preprocess(raw_features)

        # Get the outcome prediction
        prediction = self.model.predict(
            preprocessed_features, self.model.prediction_model)

        # Clip the prediction for numerical stability
        prediction = max(1e-6, min(prediction, 1-1e-6))

        return self.reverse(prediction)

    def compute_gradient(
            self,
            dose):
        """
        Compute the gradient vector.

        Parameters
        ----------
        dose : tuple
            Tuple with the dose arrays.

        Returns
        -------
        ndarray
            Gradient vector.
        """

        # Compute the feature vector
        raw_features = self.model.featurize(dose, self.segment)

        # Preprocess the feature vector
        preprocessed_features = self.model.preprocess(raw_features)

        # Get the model coefficients
        coefficients = array(self.parameter_value)

        # Get the outcome prediction
        prediction = self.model.predict(preprocessed_features)

        # Clip the prediction for numerical stability
        prediction = max(1e-6, min(prediction, 1-1e-6))

        # Get the sign
        sign = (-1)**(self.outcome_type == 'TCP')

        # Check if the transformation should be applied
        if self.transform and sign*prediction > sign*0.5:

            # Get the transformed model gradient
            model_gradient = sign*0.25*array(coefficients)

        else:

            # Get the model gradient
            model_gradient = sign*(prediction - prediction**2) * coefficients

        # Compute the preprocessing pipeline gradient
        preprocessing_gradient = (
            self.model.preprocessor.gradientize(raw_features))

        # Compute the feature gradient
        feature_gradient = self.model.gradientize(dose, self.segment)

        return (model_gradient * preprocessing_gradient) @ feature_gradient
