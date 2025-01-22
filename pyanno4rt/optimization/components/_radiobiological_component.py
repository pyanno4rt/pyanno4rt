"""Radiobiological component template."""

# Author: Tim Ortkamp

# %% External package import

from abc import ABCMeta, abstractmethod

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import compare_dictionaries

# %% Class definition


class RadiobiologicalComponent(metaclass=ABCMeta):
    """
    Radiobiological component template class.

    Parameters
    ----------
    name : str
        Name of the component class.

    segment : str
        Name of the segment associated with the component.

    parameter_name : tuple
        Name of the component parameters.

    parameter_category : tuple
        Category of the component parameters.

    parameter_value : tuple
        Value of the component parameters.

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

    parameter_name : tuple
        See 'Parameters'.

    parameter_category : tuple
        See 'Parameters'.

    parameter_value : list
        See 'Parameters'.

    embedding : {'active', 'passive'}
        See 'Parameters'.

    weight : float
        See 'Parameters'.

    rank : int
        See 'Parameters'.

    bounds : list
        See 'Parameters'.

    link : list
        See 'Parameters'.

    identifier : None or str
        See 'Parameters'.

    display : bool
        See 'Parameters'.

    adjusted_parameters : bool
        Indicator for the adjustment of the parameters due to fractionation.
    """

    def __init__(
            self,
            name,
            segment,
            parameter_name,
            parameter_category,
            parameter_value,
            embedding,
            weight,
            rank,
            bounds,
            link,
            identifier,
            display):

        # Get the class arguments
        class_arguments = locals()

        # Loop over non-required local keys
        for key in ('self', 'parameter_value'):

            # Remove the key from the class arguments dictionary
            class_arguments.pop(key)

        # Initialize the datahub
        hub = Datahub()

        # Check the class attributes
        hub.input_checker.approve(class_arguments)

        # Check the component parameters
        hub.input_checker.approve(dict(zip(parameter_name, parameter_value)))

        # Set the instance attributes from the class arguments
        self.name = name
        self.segment = segment
        self.parameter_name = parameter_name
        self.parameter_category = parameter_category
        self.parameter_value = list(parameter_value)
        self.embedding = embedding
        self.weight = float(weight)
        self.rank = rank
        self.bounds = [0.0, 1.0] if bounds is None else bounds
        self.link = [] if link is None else link
        self.identifier = identifier
        self.display = display

        # Initialize the adjustment indicator
        self.adjusted_parameters = False

    def __eq__(self, other):
        """Compare an instance with another object."""

        return (all(self.__dict__[key] == other.__dict__[key]
                    for key in ('name', 'link', 'identifier'))
                and compare_dictionaries(
                    self.__dict__.get('model_parameters', {}),
                    other.__dict__.get('model_parameters', {})))

    def get_class(self):
        """
        Get the name of the component class.

        Returns
        -------
        str
            Name of the component class.
        """

        return 'RadiobiologicalComponent'

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
            *args):
        """
        Set the value of the parameters.

        Parameters
        ----------
        *args : tuple
            Keyworded parameters. args[0] should give the value to be set.
        """

        self.parameter_value = args[0]

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
           *args):
        """
        Set the value of the weight.

        Parameters
        ----------
        *args : tuple
            Keyworded parameters. args[0] should give the value to be set.
        """

        self.weight = args[0]

    @abstractmethod
    def compute_value(
            self,
            *args):
        """Compute the component value."""

    @abstractmethod
    def compute_gradient(
            self,
            *args):
        """Compute the component gradient."""
