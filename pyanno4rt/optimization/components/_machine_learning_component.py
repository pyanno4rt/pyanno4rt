"""Machine learning component template."""

# Author: Tim Ortkamp

# %% External package import

from abc import ABCMeta, abstractmethod
from functools import partial

# %% Internal package import

from pyanno4rt.learning.models import LogisticRegression
from pyanno4rt.tools import filter_dict, wrap
from pyanno4rt.validation import (
    validate_item, validate_item_in_set, validate_length, validate_subtype,
    validate_type)

# %% Class definition


class MachineLearningComponent(metaclass=ABCMeta):
    """
    Machine learning component template class.

    Parameters
    ----------
    name : str
        Name of the component class.

    segment : str or list
        Segment(s) associated with the component.

    outcome_type : {'NTCP', 'TCP'}
        Type of the outcome variable.

    component_type : {'constraint', 'objective'}
        Type of the component.

    parameter_name : tuple
        Name of the component parameters.

    parameter_category : tuple
        Category of the component parameters.

    model : object of class \
        :class:`~pyanno4rt.learning._models._decision_tree.DecisionTree`\
        :class:`~pyanno4rt.learning._models._k_nearest_neighbors.KNearestNeighbors`\
        :class:`~pyanno4rt.learning._models._logistic_regression.LogisticRegression`\
        :class:`~pyanno4rt.learning._models._naive_bayes.NaiveBayes`\
        :class:`~pyanno4rt.learning._models._neural_network.NeuralNetwork`\
        :class:`~pyanno4rt.learning._models._random_forest.RandomForest`\
        :class:`~pyanno4rt.learning._models._support_vector_machine.SupportVectorMachine`
        The object used to represent the outcome model.

    embedding : {'active', 'passive'}
        Mode of embedding for the component. In 'passive' mode, the component \
        value is computed and tracked, but not considered in the optimization \
        problem, unlike in 'active' mode.

    weight : int or float
        Weight of the component function.

    rank : int, default=1
        Rank of the component in the lexicographic order.

    bounds : None or list
        Constraint bounds for the component.

    transform : bool, default=False
        Indicator for the transformation of the outcome function.

    identifier : None or str
        Additional string for naming the component.

    Attributes
    ----------
    name : str
        See 'Parameters'.

    segment : list
        See 'Parameters'.

    outcome_type : {'NTCP', 'TCP'}
        See 'Parameters'.

    component_type : {'constraint', 'objective'}
        See 'Parameters'.

    parameter_name : tuple
        See 'Parameters'.

    parameter_category : tuple
        See 'Parameters'.

    parameter_value : list
        Value of the component parameters.

    model : object of class \
        :class:`~pyanno4rt.learning._models._decision_tree.DecisionTree`\
        :class:`~pyanno4rt.learning._models._k_nearest_neighbors.KNearestNeighbors`\
        :class:`~pyanno4rt.learning._models._logistic_regression.LogisticRegression`\
        :class:`~pyanno4rt.learning._models._naive_bayes.NaiveBayes`\
        :class:`~pyanno4rt.learning._models._neural_network.NeuralNetwork`\
        :class:`~pyanno4rt.learning._models._random_forest.RandomForest`\
        :class:`~pyanno4rt.learning._models._support_vector_machine.SupportVectorMachine`
        See 'Parameters'.

    embedding : {'active', 'passive'}
        See 'Parameters'.

    weight : float
        See 'Parameters'.

    rank : int
        See 'Parameters'.

    bounds : list
        See 'Parameters'.

    transform : bool
        See 'Parameters'.

    identifier : None or str
        See 'Parameters'.

    adjusted_parameters : bool
        Indicator for the adjustment of the parameters due to fractionation.

    track_id : str
        Component identifier in the optimization problem tracker.

    indices : list
        Indices of the segment(s).
    """

    def __init__(
            self,
            name,
            segment,
            outcome_type,
            component_type,
            parameter_name,
            parameter_category,
            model,
            embedding,
            weight,
            rank,
            bounds,
            transform,
            identifier):

        # Validate the input arguments
        self.validate(filter_dict(locals(), remove_keys=('self',)))

        # Get the instance attributes
        self.name = name
        self.segment = wrap(segment, dtype='list')
        self.outcome_type = outcome_type
        self.component_type = component_type
        self.parameter_name = parameter_name
        self.parameter_category = parameter_category
        self.parameter_value = []
        self.model = model
        self.embedding = embedding
        self.weight = float(weight)
        self.rank = rank
        self.bounds = self.convert_bounds(bounds, embedding)
        self.transform = transform
        self.identifier = identifier

        # Initialize the adjustment indicator
        self.adjusted_parameters = False

        # Initialize the tracker identifier
        self.track_id = '-'.join(filter(
            None, (str(self.segment), self.name, self.identifier)))

        # Initialize the segment indices
        self.indices = None

    def __eq__(
            self,
            other):
        """
        Compare an instance with another object.

        Parameters
        ----------
        other : object
            The object to compare the instance with.

        Returns
        -------
        bool
            Indicator for the equality of the objects.
        """

        return all(self.__dict__[key] == other.__dict__[key] for key in (
            'name', 'segment', 'component_type', 'transform', 'identifier'))

    def __hash__(self):
        """Return the hash value."""

        return hash(
            (self.name, tuple(self.segment), self.component_type,
             self.transform, self.identifier))

    def get_class(self):
        """
        Get the name of the component class.

        Returns
        -------
        str
            Name of the component class.
        """

        return 'MachineLearningComponent'

    def convert_bounds(
            self,
            bounds,
            embedding):
        """
        Convert the bounds to function bounds.

        Parameters
        ----------
        bounds : None or list
            Constraint bounds for the component.

        embedding : {'active', 'passive'}
            Mode of embedding for the component.

        Returns
        -------
        list
            Lower and upper function bounds.
        """

        # Check if the bounds are None
        if bounds is None or embedding == 'passive':

            # Return the default function bounds
            return (0.0, 1.0)

        # Return the transformed function bounds
        return sorted(
            (0.0 if bounds[0] is None or bounds[0] < 0 else float(bounds[0]),
             1.0 if bounds[1] is None or bounds[1] > 1 else float(bounds[1])))

    def validate(
            self,
            inputs):
        """
        Validate the input arguments.

        Parameters
        ----------
        inputs : dict
            Dictionary with the mappings between argument names and values.
        """

        # Get the validation map
        validation_map = {
            'name': (
                partial(validate_type, options=str),
                ),
            'segment': (
                partial(validate_type, options=(str, list)),
                ),
            'outcome_type': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=('NTCP', 'TCP'))
                ),
            'component_type': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=(
                    'constraint', 'objective'))
                ),
            'parameter_name': (
                partial(validate_type, options=tuple),
                partial(validate_subtype, options=str)
                ),
            'parameter_category': (
                partial(validate_type, options=tuple),
                partial(validate_subtype, options=str)
                ),
            'model': (
                partial(validate_type, options=(LogisticRegression,)),
                ),
            'embedding': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=('active', 'passive'))
                ),
            'weight': (
                partial(validate_type, options=(int, float)),
                partial(validate_item, reference=0, sign='>')
                ),
            'rank': (
                partial(validate_type, options=int),
                partial(validate_item, reference=0, sign='>')
                ),
            'bounds': (
                partial(validate_type, options=(type(None), list)),
                partial(validate_length, reference=2, sign='=='),
                partial(validate_subtype, options=(type(None), int, float))
                ),
            'transform': (
                partial(validate_type, options=bool),
                ),
            'identifier': (
                partial(validate_type, options=(type(None), str)),
                )
            }

        # Loop over the dictionary keys
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)

    @abstractmethod
    def to_dict(self):
        """Serialize the component into a dictionary."""

    @classmethod
    @abstractmethod
    def from_dict(
            cls,
            dictionary):
        """Deserialize the component from a dictionary."""

    @abstractmethod
    def translate(
            self,
            value):
        """Translate function values to outcome values."""

    @abstractmethod
    def reverse(
            self,
            value):
        """Reverse outcome values to function values."""

    @abstractmethod
    def compute_value(
            self,
            dose):
        """Compute the component value."""

    @abstractmethod
    def compute_gradient(
            self,
            dose):
        """Compute the component gradient."""
