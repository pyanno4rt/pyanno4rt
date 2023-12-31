"""Objective template."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from abc import ABCMeta, abstractmethod

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Class definition


class ObjectiveClass(metaclass=ABCMeta):
    """Objective template class."""

    def __init__(
            self,
            name,
            parameter_name,
            parameter_category,
            embedding,
            weight,
            link,
            identifier,
            display):

        # Get the class arguments
        class_arguments = locals()

        # Remove the 'self'-key from the class arguments
        class_arguments.pop('self')

        # Check the class attributes and objective parameters
        Datahub().input_checker.approve(class_arguments)

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

        # Indicate the model dependency of the objective
        self.DEPENDS_ON_MODEL = False

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
