"""Target coverage initialization."""

# Author: Tim Ortkamp

# %% External package import

from numpy import hstack, ones

# %% Internal package import

from pyanno4rt.logging import get_logger
from pyanno4rt.tools import (
    flatten, get_constraint_segments, get_objective_segments)

# %% Class definition


class TargetCoverageInitializer():
    """
    Target coverage initialization class.

    This class provides methods to initialize the fluence vector with respect \
    to target coverage.

    Parameters
    ----------
    initial_fluence_vector: None or list
        User-defined initial fluence vector.

    Attributes
    ----------
    initial_fluence_vector : None or list
        See 'Parameters'.
    """

    def __init__(
            self,
            initial_fluence_vector=None):

        # Log a message about the initialization of the class
        get_logger().info("Initializing target coverage initializer ...")

        # Get the initial fluence from the argument
        self.initial_fluence_vector = initial_fluence_vector

    def run(
            self,
            segmentation,
            rbe,
            number_of_fractions,
            dose_influence_matrix,
            degrees_of_freedom):
        """
        Initialize the fluence vector with respect to target coverage.

        Returns
        -------
        ndarray
            Initial fluence vector.
        """

        # Log a message about the initialization
        get_logger().info(
            "Initializing fluence vector with respect to target coverage ...")

        def get_dose_parameters(target):
            """Get the dose-related component parameters from a target."""

            # Get the components from the target
            target_component = filter(None, flatten(
                segmentation[target][key]
                for key in ('constraint', 'objective')))

            # Return the dose parameters from all components
            return (tuple(
                component.parameter_value[index]
                for index, category in enumerate(component.parameter_category)
                if category == 'dose') for component in target_component)

        # Get the component-assigned target segments
        targets = set(
            segment for segment in (
                get_constraint_segments(segmentation)
                + get_objective_segments(segmentation))
            if segmentation[segment]['type'] == 'TARGET')

        # Check if any component-assigned target segments are present
        if len(targets) > 0:

            # Get the resized indices of the target segments
            indices = hstack([
                segmentation[target]['resized_indices'] for target in targets])

            # Get the target dose parameters
            target_doses = tuple(flatten(map(get_dose_parameters, targets)))

            # Get the maximum target dose parameter
            max_dose = max(target_doses)

        else:

            # Log a message about non-defined target components
            get_logger().info(
                "No target objectives defined - falling back to virtual "
                "target with total dose prescription of 60 Gy ...")

            # Get the resized indices of all target segments
            indices = hstack([
                segmentation[segment]['resized_indices']
                for segment in segmentation
                if segmentation[segment]['type'] == 'TARGET'])

            # Set the maximum target dose parameter for a total dose of 60 Gy
            max_dose = 60/number_of_fractions

        # Initialize a vector of ones
        ones_vector = ones((degrees_of_freedom,))

        return ones_vector * max_dose/(
            rbe * dose_influence_matrix[indices, :] @ ones_vector).mean()
