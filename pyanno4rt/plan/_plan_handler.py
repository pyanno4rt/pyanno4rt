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

    def split_components(
            self,
            verbose=True):
        """
        Split the components into objectives and constraints.

        Parameters
        ----------
        verbose : bool
            Indicator for logging output messages.

        Returns
        -------
        list
            Plan objectives.

        list
            Plan constraints.
        """

        # Get the unique components
        components = set(self.components)

        # Check if any components have been removed
        if len(components) < len(self.components) and verbose:

            # Log a message about a duplicate component
            get_logger().warning(
                "Found duplicate objects in the plan components - consider "
                "using the identifier attribute ...")

        # Get the plan objectives
        objectives = [
            component for component in components
            if component.component_type == 'objective']

        # Get the plan constraints
        constraints = list(components - set(objectives))

        return objectives, constraints

    def set_components(
            self,
            segmentation,
            verbose=True):
        """
        Set the components for plan optimization.

        Parameters
        ----------
        segmentation : dict
            Dictionary with information on the segments.

        verbose : bool
            Indicator for logging output messages.
        """

        # Loop over the segments
        for segment in segmentation:

            # Reset the objective and constraint
            segmentation[segment]['objective'] = None
            segmentation[segment]['constraint'] = None

        # Check if output messages should be logged
        if verbose:

            # Log a message about setting the components
            get_logger().info("Setting objectives and constraints ...")

        # Get the unique objectives and constraints
        objectives, constraints = self.split_components(verbose)

        # Loop over the components
        for component in objectives + constraints:

            # Get the component segment and category
            segment, category = component.segment, component.component_type

            # Check if output messages should be logged
            if verbose:

                # Log a message about setting the component
                get_logger().info(
                    "Setting %s '%s' for %s ...",
                    category, component.name, [segment]+component.link)

            # Check if the segment has no component assigned yet
            if segmentation[segment][category] is None:

                # Assign the component
                segmentation[segment][category] = component

            else:

                # Ensure the segment holds a list
                segmentation[segment][category] = wrap(
                    segmentation[segment][category], dtype='list')

                # Append the component
                segmentation[segment][category].append(component)

    def adjust_parameters_for_fractionation(
            self,
            number_of_fractions):
        """
        Adjust the dose parameters according to the number of fractions.

        Parameters
        ----------
        number_of_fractions : int
            Number of fractions according to the treatment scheme.
        """

        def adjust_component(component):
            """Adjust the dose parameters for a component."""

            # Get the component parameters
            parameters = component.get_parameter_value()

            # Loop over the indices of the dose-related parameter values
            for index in (index for index, category in enumerate(
                    component.parameter_category) if category == 'dose'):

                # Adjust the indexed parameters by the number of fractions
                parameters[index] /= number_of_fractions

            # Set the adjusted objective parameters
            component.set_parameter_value(parameters)

            # Activate the adjustment indicator of the component
            component.adjusted_parameters = True

        # Log a message about the parameter adjustment
        get_logger().info("Adjusting dose parameters for fractionation ...")

        # Adjust all non-adjusted components with dose-related parameters
        apply(adjust_component, (
            component for component in self.components
            if not component.adjusted_parameters
            and 'dose' in component.parameter_category))
