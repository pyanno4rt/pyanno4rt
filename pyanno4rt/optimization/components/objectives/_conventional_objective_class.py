"""Conventional objective template."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from abc import ABCMeta, abstractmethod

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Class definition


class ConventionalObjectiveClass(metaclass=ABCMeta):
    """Conventional objective template class."""

    def __init__(
            self,
            name,
            parameter_name,
            parameter_category,
            parameter_value,
            embedding,
            weight,
            link,
            identifier,
            display):

        # Get the class arguments
        class_arguments = locals()

        # Loop over non-required local keys
        for key in ('self', 'parameter_value'):

            # Remove the key from the locals dictionary
            class_arguments.pop(key)

        # Initialize the datahub
        hub = Datahub()

        # Check the class attributes
        hub.input_checker.approve(class_arguments)

        # Check the objective parameters
        hub.input_checker.approve(dict(zip(parameter_name, parameter_value)))

        # Set the instance attributes from the valid arguments
        self.name = name
        self.parameter_name = parameter_name
        self.parameter_category = parameter_category
        self.embedding = embedding
        self.weight = float(weight)
        self.link = [] if link is None else link
        self.identifier = identifier
        self.display = display

        # Initialize the adjustment indicator
        self.adjusted_parameters = False

        # Set the objective flags
        self.RETURNS_OUTCOME = False
        self.DEPENDS_ON_DATA = False

    def get_parameter_value(self):
        """
        Get the value of the parameters.

        Returns
        -------
        tuple
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
        args : tuple
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
        args : tuple
            Keyworded parameters. args[0] should give the value to be set.
        """
        self.weight = args[0]

    @abstractmethod
    def compute_objective_value(
            self,
            *args):
        """Compute the value of the objective function."""

    @abstractmethod
    def compute_gradient_value(
            self,
            *args):
        """Compute the value of the gradient."""
