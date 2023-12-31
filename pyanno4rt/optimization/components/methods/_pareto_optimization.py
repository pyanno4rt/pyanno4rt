"""Pareto problem."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import get_model_objectives

# %% Class definition


class ParetoOptimization():
    """
    Pareto problem class.

    This class provides methods to implement the Pareto problem for \
    solving the multi-criteria fluence optimization problem. It features the \
    tracking dictionary and computation functions for the objective and the \
    constraints.

    Parameters
    ----------
    backprojection : object of class `DoseProjection` or \
        `ConstantRBEProjection`
        Instance of the class `DoseProjection` or `ConstantRBEProjection`, \
        which inherits from `BackProjection` and provides methods to either \
        compute the dose from the fluence, or the fluence gradient from the \
        dose gradient.

    objectives : tuple
        Tuple with pairs of segmented structures and their associated \
        objectives.

    constraints : tuple
        Tuple with pairs of segmented structures and their associated \
        constraints.

    Attributes
    ----------
    name : string
        Label for the optimization problem.

    backprojection : object of class `DoseProjection` or \
        `ConstantRBEProjection`
        See 'Parameters'.

    objectives : tuple
        See 'Parameters'.

    constraints : tuple
        See 'Parameters'.
    """

    # Specify the class label
    name = 'pareto'

    def __init__(
            self,
            backprojection,
            objectives,
            constraints):

        # Log a message about the initialization of the class
        Datahub().logger.display_info("Initializing Pareto optimization "
                                      "problem ...")

        # Get the instance attributes from the arguments
        self.backprojection = backprojection
        self.objectives = objectives
        self.constraints = constraints

    def objective(
            self,
            fluence):
        """
        Compute the Pareto objective function value.

        Parameters
        ----------
        fluence : ndarray
            Values of the fluence.

        Returns
        -------
        objective_vector : list
            List of the objective function values.
        """
        # Initialize the datahub
        hub = Datahub()

        # Get the model-dependent objectives
        model_objectives = get_model_objectives(hub.segmentation)

        # Check if model-dependent objectives are present
        if len(model_objectives) > 0:

            # Loop over the model-dependent objective
            for objective in model_objectives:

                # Get the data model handler of the objective
                data_model_handler = objective.data_model_handler

                # Increment the feature calculator iteration
                data_model_handler.feature_calculator.__iteration__[1] += 1

        # Compute the dose from the fluence
        dose = self.backprojection.compute_dose(fluence)

        def compute_single_objective(objective):
            """Compute the value of a single objective function."""
            # Get the associated segments and the objective class
            segments = objective[0]
            objective_class = objective[1]

            # Check if the objective class is valid
            if all(base in str(type(objective_class).__bases__[0])
                   for base in ('pyanno4rt', 'ObjectiveClass')):

                # Call the computation function of the objective class
                objective_value = objective_class.compute_objective_value(
                    tuple(dose[hub.segmentation[segment]['resized_indices']]
                          for segment in segments), segments)

                # Check if the objective class is active
                if objective_class.embedding == 'active':

                    # Return the value of the objective function
                    return objective_value

                # Otherwise, return zero
                return 0.0

            # Raise an error if the objective class is not valid
            raise TypeError("All objective functions must have 'pyanno4rt' "
                            "and 'ObjectiveClass' in the class name, got "
                            "{} instead!"
                            .format(type(objective)))

        # Compute the objective vector
        objective_vector = [compute_single_objective(objective)
                            for objective in self.objectives]

        return objective_vector

    def constraint(
            self,
            fluence):
        """
        Compute the Pareto constraint vector.

        Parameters
        ----------
        fluence : ndarray
            Values of the fluence.

        Returns
        -------
        constraint_vector : list
            List of the constraint function values.
        """
        # Compute the dose from the fluence
        dose = self.backprojection.compute_dose(fluence)

        def compute_single_constraint(constraint):
            """Compute the value of a single constraint function."""
            # Get the associated segments and the constraint class
            segments = constraint[0]
            constraint_class = constraint[1]

            # Check if the constraint class is valid
            if all(base in str(type(constraint_class).__bases__[0])
                   for base in ('pyanno4rt', 'ConstraintClass')):

                # Check if the constraint class is active
                if constraint_class.embedding == 'active':

                    # Return the value of the constraint function
                    return constraint_class.compute_constraint(
                        tuple(
                            dose[Datahub().segmentation[
                                segment]['resized_indices']]
                            for segment in segments),
                        segments)

                # Otherwise, return zero
                return 0.0

            # Raise an error if the constraint class is not valid
            raise TypeError("All constraint functions must have 'pyanno4rt' "
                            "and 'ConstraintClass' in the class name, got "
                            "{} instead!"
                            .format(type(constraint)))

        # Compute the constraint vector
        constraint_vector = [compute_single_constraint(constraint)
                             for constraint in self.constraints]

        return constraint_vector
