"""Plan handling."""

# Author: Tim Ortkamp

# %% External package import

from math import inf

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.logging import get_logger

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

    components : dict
        See 'Parameters'.

    plan_configuration : dict
        Dictionary with information on the plan.
    """

    def __init__(
            self,
            modality,
            components):

        # Log a message about the initialization of the class
        get_logger().info("Initializing plan handler ...")

        # Get the input attributes
        self.modality = modality
        self.components = components

        # Initialize the plan configuration dictionary
        self.plan_configuration = {}

    def generate(
            self,
            segmentation):
        """
        Generate the plan configuration.

        Parameters
        ----------
        segmentation : dict
            Dictionary with information on the segments.
        """

        # Log a message about the plan generation
        get_logger().info(
            "Generating plan configuration for %s treatment ...",
            self.modality)

        # Update the plan configuration
        self.plan_configuration |= {
            'modality': self.modality,
            'RBE': 1.0 + 0.1*(self.modality == 'proton'),
            'components': self.components}

        # Store the plan configuration
        Datahub().plan_configuration = self.plan_configuration

        # Set the optimization components
        self.set_optimization_components(segmentation)

    def set_optimization_components(
            self,
            segmentation,
            verbose=True):
        """
        Set the components of the optimization problem.

        Parameters
        ----------
        segmentation : dict
            Dictionary with information on the segments.

        verbose : bool
            Indicator for logging output messages.
        """

        def set_component(component, segment, category, base):
            """Set the component by its segment and type assignment."""

            # Check if verbose is True
            if verbose:

                # Log a message about setting the component
                get_logger().info(
                    "Setting %s '%s' for %s ...",
                    category, component.name, [segment]+component.link)

            # Check if the component is already included in the base dictionary
            if component.track_id not in base:

                # Add the instance to the base dictionary
                base[component.track_id] = {
                    'segments': [segment]+component.link,
                    'instance': component}

                # Check if no instance has been set yet
                if not segmentation[segment][category]:

                    # Add the instance to the segment
                    segmentation[segment][category] = component

                else:

                    # Make a list and add the instance
                    segmentation[segment][category] = [
                        segmentation[segment][category], component]

        # Initialize the datahub
        hub = Datahub()

        # Loop over the segments
        for segment in segmentation:

            # Reset the segment objective and constraint key
            segmentation[segment]['objective'] = None
            segmentation[segment]['constraint'] = None

        # Check if verbose is True
        if verbose:

            # Log a message about the components setting
            get_logger().info("Setting objectives and constraints ...")

        # Initialize the objective and constraint dictionaries
        objectives, constraints = {}, {}

        # Set the base dictionaries for the component types
        bases = {'objective': objectives, 'constraint': constraints}

        # Loop over the segments in the components dictionary
        for component in self.components:

            # Get the segment and component type
            segment, category = component.segment, component.component_type

            # Get the base dictionary
            base = bases[category]

            # Set the component
            set_component(component, segment, category, base)

        # Loop over the constraints
        for constraint in constraints.values():

            # Get the constraint instance
            instance = constraint['instance']

            # Overwrite the constraint bounds by the embedding type
            instance.bounds = (
                instance.bounds if instance.embedding == 'active'
                else [-inf, inf])

        # Check if the optimization dictionary already exists
        if hub.optimization:

            # Add the objectives and constraints to the datahub
            hub.optimization |= {
                'objectives': objectives,
                'constraints': constraints}

        else:

            # Enter the objectives and constraints into the datahub
            hub.optimization = {
                'objectives': objectives,
                'constraints': constraints}
