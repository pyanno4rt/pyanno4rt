"""Support vector machine NTCP component."""

# Author: Tim Ortkamp

# %% External package import

from copy import deepcopy

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning import DataModelHandler, ModelParameters
from pyanno4rt.learning.svm import (
    linear_decision_function, poly_decision_function, rbf_decision_function,
    sigmoid_decision_function, linear_decision_gradient,
    poly_decision_gradient, rbf_decision_gradient, sigmoid_decision_gradient,
    SupportVectorMachineModel)
from pyanno4rt.optimization.components import MachineLearningComponent
from pyanno4rt.tools import filter_dict, inverse_sigmoid

# %% Class definition


class SupportVectorMachineNTCP(MachineLearningComponent):
    """
    Support vector machine NTCP component class.

    This class provides methods to compute the value and the gradient of the \
    support vector machine NTCP component, as well as to add the support \
    vector machine model.

    Parameters
    ----------
    segment : str
        Name of the segment associated with the component.

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

    identifier : None or str, default=None
        Additional string for naming the component.

    display : bool, default=True
        Indicator for the display of the component.

    Attributes
    ----------
    arguments : dict
        Dictionary with the component input arguments (for serialization).

    decision_function : None or callable
        Decision function for the fitted kernel type.

    decision_gradient : None or callable
        Decision gradient for the fitted kernel type.

    data_model_handler : object of class \
        :class:`~pyanno4rt.learning._data_model_handler.DataModelHandler`
        The object used to handle the dataset, the feature map generation and \
        the feature (re-)calculation.

    model : object of class \
        :class:`~pyanno4rt.learning.svm._support_vector_machine.SupportVectorMachineModel`
        The object used to preprocess, tune, train, inspect and evaluate the \
        support vector machine model.

    parameter_value : list
        Primal/dual support vector machine model coefficients.

    bounds : list
        See 'Parameters'. Transformed by the inverse Platt scaling function.
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
            name='Support Vector Machine NTCP',
            segment=segment,
            component_type=component_type,
            parameter_name=('w/alpha',),
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

        # Initialize the decision function/gradient
        self.decision_function, self.decision_gradient = None, None

    def to_dict(self):
        """Serialize the component into a dictionary."""

        # Get the parameter dictionary
        dictionary = deepcopy(self.arguments)

        # Serialize the model parameters
        dictionary['model_parameters'] = (
            dictionary['model_parameters'].to_dict())

        return {'Support Vector Machine NTCP': dictionary}

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
            :class:`~pyanno4rt.optimization.components._support_vector_machine_ntcp.SupportVectorMachineNTCP`
            The object used to handle the component parameters.
        """

        # Deserialize the model parameters
        dictionary['model_parameters'] = ModelParameters.from_dict(
            dictionary['model_parameters'])

        return cls(**dictionary)

    def add_model(self):
        """Add the support vector machine model to the component."""

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the model addition
        hub.logger.display_info(
            f"Adding support vector machine model for '{self.name}' ...")

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

        # Initialize the support vector machine model
        self.model = SupportVectorMachineModel(
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

        # Check if the linear kernel has been fitted
        if self.model.prediction_model.kernel == 'linear':

            # Get the primal coefficients
            self.parameter_value = (
                self.model.prediction_model.coef_[0].tolist())

        else:

            # Get the dual coefficients
            self.parameter_value = (
                self.model.prediction_model.dual_coef_[0].tolist())

        # Map the kernel types to the decision functions/gradients
        decision_map = {
            'linear': (linear_decision_function, linear_decision_gradient),
            'poly': (poly_decision_function, poly_decision_gradient),
            'rbf': (rbf_decision_function, rbf_decision_gradient),
            'sigmoid': (sigmoid_decision_function, sigmoid_decision_gradient)}

        # Get the decision function/gradient
        self.decision_function, self.decision_gradient = (
            decision_map[self.model.prediction_model.kernel])

        # Transform the component bounds
        self.bounds = sorted(inverse_sigmoid(
            bound, -self.model.prediction_model.probA_[0],
            self.model.prediction_model.probB_[0]) for bound in self.bounds)

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

        return self.decision_function(
            self.model.prediction_model, preprocessed_features)

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
        model_gradient = self.decision_gradient(
            self.model.prediction_model, preprocessed_features)

        # Compute the preprocessing pipeline gradient
        preprocessing_gradient = (
            self.model.preprocessor.gradientize(raw_features))

        # Compute the feature gradient
        feature_gradient = feature_calculator.gradientize(dose, segment)

        return (model_gradient * preprocessing_gradient) @ feature_gradient
