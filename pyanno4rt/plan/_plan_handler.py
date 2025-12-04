"""Plan handling."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.logging import get_logger
from pyanno4rt.tools import apply, wrap

# %% Class definition


class PlanHandler():
    """
    Plan handling class.

    This class provides methods to handle plan data.

    Parameters
    ----------
    modality : {'photon', 'proton'}
        Treatment modality.

    components : dict
        Optimization components for each segment of interest, i.e., \
        objective functions and constraints.

    Attributes
    ----------
    modality : {'photon', 'proton'}
        See 'Parameters'.

    RBE : float
        Relative biological effectiveness, depending on the modality.

    components : dict
        See 'Parameters'.
    """

    def __init__(
            self,
            modality,
            components):

        # Log a message about the initialization of the class
        get_logger().info("Initializing plan handler ...")

        # Get the input attributes
        self.modality = modality
        self.RBE = 1.0 + 0.1*(modality == 'proton')
        self.components = components

    def index_components(
            self,
            segmentation):
        """
        Add the segment indices to the components.

        Parameters
        ----------
        segmentation : dict
            Dictionary with information on the segments.
        """

        # Log a message about adding the segment indices
        get_logger().info("Adding segment indices to the components ...")

        # Loop over the components
        for component in self.components:

            # Store the resized indices
            component.indices = [
                segmentation[segment]['resized_indices']
                for segment in wrap(component.segment)]

    def fractionate_components(
            self,
            number_of_fractions):
        """
        Adjust the components for fractionation.

        Parameters
        ----------
        number_of_fractions : int
            Number of fractions according to the treatment scheme.
        """

        def adjust_component(component):
            """Adjust the parameters for a component."""

            # Get the component parameters
            parameters = component.parameter_value

            # Loop over the indices of the dose-related parameter values
            for index in (index for index, category in enumerate(
                    component.parameter_category) if category == 'dose'):

                # Adjust the indexed parameters by the number of fractions
                parameters[index] /= number_of_fractions

            # Set the adjustment indicator
            component.adjusted_parameters = True

        # Log a message about the fractionation
        get_logger().info("Adjusting components for fractionation ...")

        # Adjust the non-fractionated components
        apply(adjust_component, (
            component for component in self.components
            if not component.adjusted_parameters
            and 'dose' in component.parameter_category))
