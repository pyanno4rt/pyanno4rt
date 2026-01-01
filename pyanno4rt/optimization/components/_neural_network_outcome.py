"""Neural network outcome component."""

# Author: Tim Ortkamp

# %% External package import

from copy import deepcopy

# %% Internal package import

from pyanno4rt.learning.models import FeedForwardNet
from pyanno4rt.optimization.components import MachineLearningComponent
from pyanno4rt.tools import (
    filter_dict, inverse_salu, inverse_sigmoid, salu, sigmoid)

# %% Class definition


class NeuralNetworkOutcome(MachineLearningComponent):
    """
    Neural network outcome component class.

    This class provides methods to handle a neural network outcome \
    model-based component.

    Parameters
    ----------
    segment : str or list
        Segment(s) associated with the component.

    outcome_type : {'NTCP', 'TCP'}
        Type of the outcome variable.

    model : object of class \
        :class:`~pyanno4rt.learning.models.neural_network._feed_forward_net.FeedForwardNet`
        The object used to represent the neural network outcome model.

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
        Dictionary with the component input arguments (for serialization).

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
            name='Neural Network Outcome',
            segment=segment,
            outcome_type=outcome_type,
            component_type=component_type,
            parameter_name=('(weight, bias)',),
            parameter_category=('parameter',),
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
        """Serialize the component into a dictionary."""

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
            Dictionary with the component parameters.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.optimization.components._neural_network_outcome.NeuralNetworkOutcome`
            The object used to represent the neural network outcome component.
        """

        # Deserialize the model
        dictionary['model'] = FeedForwardNet.from_dict(dictionary['model'])

        return cls(**dictionary)

    def update_from_model(self):
        """Update the component from the outcome model."""

        # Store the model coefficients
        self.parameter_value = list(
            weight for weights in (
                layer.flatten().astype(float)
                if len(layer.shape) == 2
                else layer.astype(float)
                for layer in self.model.predictor.get_weights())
            for weight in weights)

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
            return sigmoid(inverse_salu(value, self.sign))

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
            return salu(inverse_sigmoid(value), self.sign)

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
            (self.sign*predictor_gradient * preprocessing_gradient)
            @ feature_gradient)
