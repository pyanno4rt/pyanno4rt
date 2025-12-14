"""Target coverage initialization."""

# Author: Tim Ortkamp

# %% External package import

from numpy import array, hstack, ones

# %% Internal package import

from pyanno4rt.logging import get_logger
from pyanno4rt.tools import flatten, get_all_segments

# %% Class definition


class TargetCoverageInitializer():
    """
    Target coverage initialization class.

    This class provides methods to initialize the fluence vector with respect \
    to target coverage.

    Parameters
    ----------
    initial_fluence: None or list
        Initial fluence vector.

    Attributes
    ----------
    initial_fluence : None or list
        See 'Parameters'.
    """

    def __init__(
            self,
            initial_fluence=None):

        # Log a message about the initialization of the class
        get_logger().info("Initializing target coverage strategy ...")

        # Get the initial fluence
        self.initial_fluence = initial_fluence

    def run(
            self,
            handlers):
        """
        Initialize the fluence vector with respect to target coverage.

        Parameters
        ----------
        handlers : dict
            Dictionary with the handlers (patient, plan, dose, data models).

        Returns
        -------
        ndarray
            Initial fluence vector.
        """

        def get_dose_parameters(component):
            """Get the dose parameters from a component."""

            return tuple(
                component.parameter_value[index]
                for index, category in enumerate(component.parameter_category)
                if category == 'dose')

        # Check if an initial fluence vector has been provided
        if self.initial_fluence is not None:

            # Log a message about falling back to warm-start strategy
            get_logger().warning(
                "User has provided an initial fluence vector - falling back "
                "to warm-start strategy ...")

            return array(self.initial_fluence)

        # Log a message about the initialization
        get_logger().info(
            "Initializing fluence vector with respect to target coverage ...")

        # Get the segmentation data
        segmentation = handlers['patient_handler'].segmentation

        # Get the target segments
        target_segments = [
            segment for segment in get_all_segments(
                handlers['plan_handler'].components)
            if segmentation[segment]['type'] == 'TARGET']

        # Get the target components
        target_components = [
            component for component in handlers['plan_handler'].components
            if not set(component.segment).isdisjoint(target_segments)]

        # Check if any target components are present
        if len(target_components) > 0:

            # Get the joint target indices
            indices = hstack([
                segmentation[segment]['resized_indices']
                for segment in target_segments])

            # Get the maximum prescription dose
            max_dose = max(flatten(map(
                get_dose_parameters, target_components)))

        else:

            # Log a message about non-defined target components
            get_logger().warning(
                "No target objectives defined - falling back to virtual "
                "target with total dose prescription of 60 Gy ...")

            # Get the indices of all target segments
            indices = hstack([
                segmentation[segment]['resized_indices']
                for segment in segmentation
                if segmentation[segment]['type'] == 'TARGET'])

            # Set the maximum prescription dose to 60 Gy
            max_dose = 60/handlers['dose_handler'].number_of_fractions

        # Initialize the unit fluence vector
        unit_fluence = ones((handlers['dose_handler'].degrees_of_freedom,))

        return unit_fluence * max_dose/(
            handlers['plan_handler'].RBE
            * handlers['dose_handler'].dose_influence_matrix[indices, :]
            @ unit_fluence).mean()
