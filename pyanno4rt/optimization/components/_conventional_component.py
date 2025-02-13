"""Conventional component template."""

# Author: Tim Ortkamp

# %% External package import

from abc import ABCMeta, abstractmethod
from math import inf

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Class definition


class ConventionalComponent(metaclass=ABCMeta):
    """
    Conventional component template class.

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

    rank : int
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

        # Check the component parameter value(s)
        hub.input_checker.approve(dict(zip(parameter_name, parameter_value)))

        # Set the instance attributes from the class arguments
        self.name = name
        self.segment = segment
        self.parameter_name = parameter_name
        self.parameter_category = parameter_category
        self.parameter_value = list(map(float, parameter_value))
        self.embedding = embedding
        self.weight = float(weight)
        self.rank = rank
        self.bounds = self.convert_bounds(bounds, embedding)
        self.link = [] if link is None else link
        self.identifier = identifier
        self.display = display

        # Initialize the adjustment indicator
        self.adjusted_parameters = False

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
            'name', 'link', 'identifier'))

    def get_class(self):
        """
        Get the name of the component class.

        Returns
        -------
        str
            Name of the component class.
        """

        return 'ConventionalComponent'

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
            return [-inf, inf]

        # Return the transformed function bounds
        return sorted(
            (-inf if bounds[0] is None else float(bounds[0]),
             inf if bounds[1] is None else float(bounds[1])))

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
    def compute_value(
            self,
            dose,
            *args):
        """Compute the component value."""

    @abstractmethod
    def compute_gradient(
            self,
            dose,
            *args):
        """Compute the component gradient."""
