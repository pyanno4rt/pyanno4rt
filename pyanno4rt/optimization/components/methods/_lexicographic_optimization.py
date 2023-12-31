"""Lexicographic problem."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import array

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import get_model_objectives

# %% Class definition


class LexicographicOptimization():
    """
    Lexicographic problem class.

    This class provides methods to implement the lexicographic method for \
    solving the scalarized fluence optimization problem. It features the \
    tracking dictionary and computation functions for the objective, the \
    gradient, and the constraints.

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
    backprojection : object of class `DoseProjection` or \
        `ConstantRBEProjection`
        See 'Parameters'.

    objectives : tuple
        See 'Parameters'.

    constraints : tuple
        See 'Parameters'.

    tracker : dict
        Dictionary with the objective function values for each iteration, \
        divided by the associated segments.
    """

    def __init__(
            self,
            backprojection,
            objectives,
            constraints):

        # Log a message about the initialization of the class
        Datahub().logger.display_info("Initializing lexicographic "
                                      "optimization problem ...")

        # Get the instance attributes from the arguments
        self.backprojection = backprojection
        self.objectives = objectives
        self.constraints = constraints

        # Initialize the tracker dictionary
        self.tracker = dict(
            {"Total": []},
            **{segment_objective: []
               for segment_objective in (
                '-'.join((str(objective[0]), objective[1].name))
                if objective[1].identifier is None
                else '-'.join((str(objective[0]), objective[1].name,
                               objective[1].identifier))
                for objective in self.objectives)})

    def objective(
            self,
            fluence):
        """
        Compute the lexicographic objective function value.

        Parameters
        ----------
        fluence : ndarray
            Values of the fluence.

        Returns
        -------
        total_objective_value : float
            Value of the weighted-sum objective function.
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

                # Enter the value into the tracking dictionary
                self.tracker[
                    '-'.join((str(segments), objective_class.name))
                    if objective_class.identifier is None
                    else '-'.join((str(segments), objective_class.name,
                                   objective_class.identifier))
                    ] += (objective_value * objective_class.weight,)

                # Check if the objective class is active
                if objective_class.embedding == 'active':

                    # Return the value of the objective function
                    return objective_value * objective_class.weight

                # Otherwise, return zero
                return 0.0

            # Raise an error if the objective class is not valid
            raise TypeError("All objective functions must have 'pyanno4rt' "
                            "and 'ObjectiveClass' in the class name, got "
                            "{} instead!"
                            .format(type(objective)))

        # Compute the total objective value
        total_objective_value = sum(compute_single_objective(objective)
                                    for objective in self.objectives)

        # Enter the total value into the tracking dictionary
        self.tracker["Total"] += (total_objective_value,)

        return total_objective_value

    def gradient(
            self,
            fluence):
        """
        Compute the lexicographic gradient vector.

        Parameters
        ----------
        fluence : ndarray
            Values of the fluence.

        Returns
        -------
        fluence_gradient : ndarray
            Values of the weighted-sum fluence derivatives.
        """
        # Compute the dose from the fluence
        dose = self.backprojection.compute_dose(fluence)

        def compute_single_gradient(objective):
            """Compute the value of a single gradient function."""
            # Get the associated segments and the objective class
            segments = objective[0]
            objective_class = objective[1]

            # Check if the objective class is valid
            if all(base in str(type(objective_class).__bases__[0])
                   for base in ('pyanno4rt', 'ObjectiveClass')):

                # Check if the objective class is active
                if objective_class.embedding == 'active':

                    # Return the value of the gradient function
                    return (objective_class.compute_gradient_value(
                        tuple(
                            dose[Datahub().segmentation[
                                segment]['resized_indices']]
                            for segment in segments), segments)
                            * objective_class.weight)

                # Otherwise, return zero
                return 0.0

            # Raise an error if the objective class is not valid
            raise TypeError("All objective functions must have 'pyanno4rt' "
                            "and 'ObjectiveClass' in the class name, got "
                            "{} instead!"
                            .format(type(objective)))

        # Compute the total gradient vector
        total_gradient = sum(compute_single_gradient(objective)
                             for objective in self.objectives)

        # Compute the fluence gradient from the dose gradient
        fluence_gradient = self.backprojection.compute_fluence_gradient(
            total_gradient)

        return fluence_gradient

    def constraint(
            self,
            fluence):
        """
        Compute the lexicographic constraint vector.

        Parameters
        ----------
        fluence : ndarray
            Values of the fluence.

        Returns
        -------
        constraints : ndarray
            Values of the constraints.
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

        # Compute the values of all constraint functions
        constraints = array(compute_single_constraint(constraint)
                            for constraint in self.constraints)

        return constraints
