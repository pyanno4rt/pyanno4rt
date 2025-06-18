"""Machine learning component template."""

# Author: Tim Ortkamp

# %% External package import

from abc import ABCMeta, abstractmethod
from functools import partial

# %% Internal package import

from pyanno4rt.checking import (
    check_length, check_subtype, check_type, check_value, check_value_in_set)
from pyanno4rt.learning import ModelParameters
from pyanno4rt.tools import compare_dictionaries, filter_dict

# %% Class definition


class MachineLearningComponent(metaclass=ABCMeta):
    """
    Machine learning component template class.

    Parameters
    ----------
    name : str
        Name of the component class.

    segment : str
        Name of the segment associated with the component.

    component_type : {'constraint', 'objective'}
        Type of the component.

    parameter_name : tuple
        Name of the component parameters.

    parameter_category : tuple
        Category of the component parameters.

    model_parameters : object of class \
        :class:`~pyanno4rt.learning._model_parameters.ModelParameters`
        The object used to represent the learning model parameters.

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

    link : None or list
        Other segments used for joint evaluation.

    identifier : None or str
        Additional string for naming the component.

    display : bool
        Indicator for the display of the component.

    Attributes
    ----------
    name : str
        See 'Parameters'.

    segment : str
        See 'Parameters'.

    component_type : {'constraint', 'objective'}
        See 'Parameters'.

    parameter_name : tuple
        See 'Parameters'.

    parameter_category : tuple
        See 'Parameters'.

    parameter_value : list
        Value of the component parameters.

    model_parameters : object of class \
        :class:`~pyanno4rt.learning._model_parameters.ModelParameters`
        See 'Parameters'.

    embedding : {'active', 'passive'}
        See 'Parameters'.

    weight : float
        See 'Parameters'.

    rank : int
        See 'Parameters'.

    bounds : list
        See 'Parameters'.

    link : None or list
        See 'Parameters'.

    identifier : None or str
        See 'Parameters'.

    display : bool
        See 'Parameters'.

    data_model_handler : None
        Initial variable for the object used to handle the dataset, the \
        feature map generation and the feature (re-)calculation.

    model : None
        Initial variable for the object used to preprocess, tune, train, \
        inspect and evaluate the machine learning model.

    adjusted_parameters : bool
        Indicator for the adjustment of the parameters due to fractionation.

    track_id : str
        Component identifier in the optimization problem tracker.
    """

    def __init__(
            self,
            name,
            segment,
            component_type,
            parameter_name,
            parameter_category,
            model_parameters,
            embedding,
            weight,
            rank,
            bounds,
            link,
            identifier,
            display):

        # Check the input arguments
        self.check(filter_dict(locals(), remove_keys=('self',)))

        # Set the instance attributes from the class arguments
        self.name = name
        self.segment = segment
        self.component_type = component_type
        self.parameter_name = parameter_name
        self.parameter_category = parameter_category
        self.parameter_value = []
        self.model_parameters = model_parameters
        self.embedding = embedding
        self.weight = float(weight)
        self.rank = rank
        self.bounds = self.convert_bounds(bounds, embedding)
        self.link = [] if link is None else link
        self.identifier = identifier
        self.display = display

        # Initialize the data model handler and the outcome model
        self.data_model_handler = None
        self.model = None

        # Initialize the adjustment indicator
        self.adjusted_parameters = False

        # Initialize the tracker identifier
        self.track_id = '-'.join(filter(
            None, (f"{[self.segment]+self.link}", self.name, self.identifier)))

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

        return (
            all(self.__dict__[key] == other.__dict__[key] for key in (
                'name', 'segment', 'component_type', 'link', 'identifier'))
            and compare_dictionaries(
                self.model_parameters.to_dict(),
                other.model_parameters.to_dict()))

    def check(
            self,
            inputs):
        """
        Check the input arguments.

        Parameters
        ----------
        inputs : dict
            Dictionary with the mappings between argument names and values.
        """

        # Get the check map
        check_map = {
            'name': (
                partial(check_type, options=str),),
            'segment': (
                partial(check_type, options=str),),
            'component_type': (
                partial(check_type, options=str),
                partial(
                    check_value_in_set, options=('constraint', 'objective'))),
            'parameter_name': (
                partial(check_type, options=tuple),
                partial(check_subtype, options=str)),
            'parameter_category': (
                partial(check_type, options=tuple),
                partial(check_subtype, options=str)),
            'model_parameters': (
                partial(check_type, options=ModelParameters),),
            'embedding': (
                partial(check_type, options=str),
                partial(check_value_in_set, options=('active', 'passive'))),
            'weight': (
                partial(check_type, options=(int, float)),
                partial(check_value, reference=0, sign='>')),
            'rank': (
                partial(check_type, options=int),
                partial(check_value, reference=0, sign='>')),
            'bounds': (
                partial(check_type, options=(type(None), list)),
                partial(check_length, reference=2, sign='=='),
                partial(check_subtype, options=(type(None), int, float))),
            'link': (
                partial(check_type, options=(type(None), list)),
                partial(check_subtype, options=str)),
            'identifier': (
                partial(check_type, options=(type(None), str)),),
            'display': (
                partial(check_type, options=bool),)}

        # Loop over the dictionary keys
        for key, value in inputs.items():

            # Loop over the check functions
            for function in check_map[key]:

                # Run the check function
                function(key, value)

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

        # Get the (N)TCP function sign
        sign = (-1.0)**('NTCP' not in self.name)

        # Check if the bounds are None
        if bounds is None or embedding == 'passive':

            # Return the default function bounds
            return sorted((0.0, sign))

        # Return the transformed function bounds
        return sorted(
            (0.0 if bounds[0] is None or bounds[0] < 0 else sign*bounds[0],
             sign if bounds[1] is None or bounds[1] > 1 else sign*bounds[1]))

    def get_parameter_value(self):
        """
        Get the value of the parameters.

        Returns
        -------
        list
            Value of the parameters.
        """

        return self.parameter_value

    def set_parameter_value(
            self,
            value):
        """
        Set the value of the parameters.

        Parameters
        ----------
        value : list
            Value to be set.
        """

        self.parameter_value = value

    def get_weight_value(self):
        """
        Get the value of the weight.

        Returns
        -------
        float
            Value of the weight.
        """

        return self.weight

    def set_weight_value(
           self,
           value):
        """
        Set the value of the weight.

        Parameters
        ----------
        value : float
            Value to be set.
        """

        self.weight = value

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
    def add_model(self):
        """Add the machine learning model to the component."""

    @abstractmethod
    def compute_value(
            self,
            dose,
            segment):
        """Compute the component value."""

    @abstractmethod
    def compute_gradient(
            self,
            dose,
            segment):
        """Compute the component gradient."""
