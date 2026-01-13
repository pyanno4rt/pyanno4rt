"""Support vector machine outcome component."""

# Author: Tim Ortkamp

# %% External package import

from copy import deepcopy

# %% Internal package import

from pyanno4rt.learning.models import SupportVectorMachine
from pyanno4rt.optimization.components import MachineLearningComponent
from pyanno4rt.tools import (
    filter_dict, inverse_salu, inverse_sigmoid, salu, sigmoid)

# %% Class definition


class SupportVectorMachineOutcome(MachineLearningComponent):
    """
    Support vector machine outcome component class.

    This class provides methods to handle a support vector machine outcome \
    model-based component.

    Parameters
    ----------
    segment : str or list
        Segment(s) associated with the component.

    outcome_type : {'NTCP', 'TCP'}
        Type of the outcome variable.

    model : object of class \
        :class:`~pyanno4rt.learning.models.svm._support_vector_machine.SupportVectorMachine`
        The object used to represent the support vector machine outcome model.

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

    transform : bool, default=False
        Indicator for the transformation of the outcome function.

    identifier : None or str, default=None
        Additional string for naming the component.

    Attributes
    ----------
    arguments : dict
        Dictionary with the input arguments (for serialization).

    Notes
    -----
    See :class:`~pyanno4rt.optimization.components._machine_learning_component.MachineLearningComponent`\
    for details on the inherited attributes.
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

        # Call the superclass constructor
        super().__init__(
            name='Support Vector Machine Outcome',
            segment=segment,
            outcome_type=outcome_type,
            component_type=component_type,
            parameter_name=('w/alpha',),
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

    def to_dict(self):
        """
        Serialize the component into a dictionary.

        Returns
        -------
        dict
            Dictionary with the component's arguments.
        """

        # Get the parameter dictionary
        dictionary = deepcopy(self.arguments)

        # Serialize the model
        dictionary['model'] = dictionary['model'].to_dict()

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
            Dictionary with the component's arguments.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.optimization.components._support_vector_machine_outcome.SupportVectorMachineOutcome`
            The object used to represent the support vector machine outcome \
            component.
        """

        # Deserialize the model
        dictionary['model'] = SupportVectorMachine.from_dict(
            dictionary['model'])

        return cls(**dictionary)

    def update_from_model(self):
        """Update the component from the outcome model."""

        # Check if the linear kernel has been fitted
        if self.model.predictor.kernel == 'linear':

            # Store the primal support vector machine coefficients
            self.parameter_value = self.model.predictor.coef_[0].tolist()

        else:

            # Store the dual support vector machine coefficients
            self.parameter_value = self.model.predictor.dual_coef_[0].tolist()

        # Update the bounds
        self.bounds = sorted(
            self.reverse(bound) for bound in self.convert_bounds(
                self.arguments['bounds'], self.embedding))

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

        # Check if the transformation should be applied
        if self.transform:

            # Return the transformed outcome value
            return sigmoid(
                inverse_salu(
                    value, self.model.multiplier, self.model.summand,
                    self.sign),
                self.model.multiplier, self.model.summand)

        # Check if the value is an iterable
        if isinstance(value, (tuple, list)):

            # Return a list of outcome values
            return [self.sign*val for val in value]

        # Return a single outcome value
        return self.sign*value

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

        # Check if the transformation should be applied
        if self.transform:

            # Return the transformed function value
            return salu(
                inverse_sigmoid(
                    value, self.model.multiplier, self.model.summand),
                self.model.multiplier, self.model.summand, self.sign)

        # Check if the value is an iterable
        if isinstance(value, (tuple, list)):

            # Return a list of function values
            return [self.sign*val for val in value]

        # Return a single function value
        return self.sign*value

    def compute_value(
            self,
            dose):
        """
        Compute the function value.

        Parameters
        ----------
        dose : tuple
            Dose vectors.

        Returns
        -------
        float
            Function value.
        """

        # Compute the feature vector
        raw_features = self.model.featurize(dose, self.segment)

        # Preprocess the feature vector
        preprocessed_features, _ = self.model.preprocess(raw_features)

        # Get the outcome prediction
        prediction = self.model.predict(preprocessed_features)

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
            Dose vectors.

        Returns
        -------
        ndarray
            Gradient vector.
        """

        # Compute the feature vector
        raw_features = self.model.featurize(dose, self.segment)

        # Preprocess the feature vector
        preprocessed_features, _ = self.model.preprocess(raw_features)

        # Get the outcome prediction
        prediction = self.model.predict(preprocessed_features)

        # Clip the prediction for numerical stability
        prediction = max(1e-6, min(prediction, 1-1e-6))

        # Get the model gradients
        feature_gradient, preprocessing_gradient, predictor_gradient = (
            self.model.gradientize(dose, self.segment))

        # Check if the transformation should be applied
        if self.transform and self.sign*prediction > self.sign*0.5:

            # Get the transformed predictor gradient
            predictor_gradient = (
                0.25*predictor_gradient/(prediction - prediction**2))

        return (
            feature_gradient
            @ preprocessing_gradient
            @ (self.sign*predictor_gradient))
