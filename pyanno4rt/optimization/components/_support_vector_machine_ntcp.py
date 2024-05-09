"""Support vector machine NTCP component."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model import DataModelHandler
from pyanno4rt.learning_model.frequentist import SupportVectorMachineModel
from pyanno4rt.learning_model.frequentist.additional_files import (
    linear_decision_function, rbf_decision_function, poly_decision_function,
    sigmoid_decision_function, linear_decision_gradient, rbf_decision_gradient,
    poly_decision_gradient, sigmoid_decision_gradient)
from pyanno4rt.optimization.components import MachineLearningComponentClass
from pyanno4rt.tools import inverse_sigmoid

# %% Class definition


class SupportVectorMachineNTCP(MachineLearningComponentClass):
    """
    Support vector machine NTCP component class.

    This class provides methods to compute the value and the gradient of the \
    support vector machine NTCP component, as well as to add the support \
    vector machine model.

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

    bounds : None or list, default=None
        Constraint bounds for the component.

    link : None or list, default=None
        Other segments used for joint evaluation.

    identifier : str, default=None
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
        :class:`~pyanno4rt.learning_model.frequentist._support_vector_machine.SupportVectorMachineModel`
        The object used to preprocess, tune, train, inspect and evaluate the \
        support vector machine model.

    parameter_value : list
        Value of the primal/dual support vector machine model coefficients.

    decision_function : callable
        Decision function for the fitted kernel type.

    decision_gradient : callable
        Decision gradient for the fitted kernel type.

    bounds : list
        See 'Parameters'. Transformed by the inverse Platt scaling function.
    """

    def __init__(
            self,
            model_parameters,
            embedding='active',
            weight=1.0,
            bounds=None,
            link=None,
            identifier=None,
            display=True):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(name='Support Vector Machine NTCP',
                         parameter_name=('w/alpha',),
                         parameter_category=('coefficient',),
                         model_parameters=model_parameters,
                         embedding=embedding,
                         weight=weight,
                         bounds=bounds,
                         link=link,
                         identifier=identifier,
                         display=display)

        # Initialize the decision function/gradient
        self.decision_function, self.decision_gradient = None, None

    def add_model(self):
        """Add the support vector machine model to the component."""

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the model addition
        hub.logger.display_info(
            f"Adding support vector machine model for '{self.name}' ...")

        # Initialize the data model handler
        self.data_model_handler = DataModelHandler(
            model_label=self.model_parameters['model_label'],
            data_path=self.model_parameters['data_path'],
            feature_filter=self.model_parameters['feature_filter'],
            label_name=self.model_parameters['label_name'],
            label_bounds=self.model_parameters['label_bounds'],
            time_variable_name=self.model_parameters['time_variable_name'],
            label_viewpoint=self.model_parameters['label_viewpoint'],
            fuzzy_matching=self.model_parameters['fuzzy_matching'],
            write_features=self.model_parameters['write_features'])

        # Integrate the model-related classes
        self.data_model_handler.integrate()

        # Initialize the support vector machine model
        self.model = SupportVectorMachineModel(
            model_label=self.model_parameters['model_label'],
            model_folder_path=self.model_parameters['model_folder_path'],
            dataset=hub.datasets[self.model_parameters['model_label']],
            preprocessing_steps=self.model_parameters['preprocessing_steps'],
            tune_space=self.model_parameters['tune_space'],
            tune_evaluations=self.model_parameters['tune_evaluations'],
            tune_score=self.model_parameters['tune_score'],
            tune_splits=self.model_parameters['tune_splits'],
            inspect_model=self.model_parameters['inspect_model'],
            evaluate_model=self.model_parameters['evaluate_model'],
            oof_splits=self.model_parameters['oof_splits'],
            display_options=self.model_parameters['display_options'])

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
            'rbf': (rbf_decision_function, rbf_decision_gradient),
            'poly': (poly_decision_function, poly_decision_gradient),
            'sigmoid': (sigmoid_decision_function, sigmoid_decision_gradient)}

        # Get the decision function/gradient
        self.decision_function, self.decision_gradient = (
            decision_map[self.model.prediction_model.kernel])

        # Transform the component bounds
        self.bounds = sorted(
            inverse_sigmoid(
                bound, -self.model.prediction_model.probA_[0],
                self.model.prediction_model.probB_[0])
            for bound in self.bounds)

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

        # Preprocess the feature vector
        preprocessed_features = self.model.preprocess(raw_features)

        return self.decision_function(
            self.model.prediction_model, preprocessed_features)

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

        # Preprocess the feature vector
        preprocessed_features = self.model.preprocess(raw_features)

        # Compute the model gradient
        model_gradient = self.decision_gradient(
            self.model.prediction_model, preprocessed_features)

        # Compute the preprocessing pipeline gradient
        preprocessing_gradient = (
            self.model.preprocessor.gradientize(raw_features))

        # Compute the feature gradient
        feature_gradient = feature_calculator.gradientize(args[0], args[1])

        return (model_gradient * preprocessing_gradient) @ feature_gradient
