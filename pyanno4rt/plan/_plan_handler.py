"""Plan handling."""

# Author: Tim Ortkamp

# %% External package import

from math import inf

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Class definition


class PlanHandler():
    """
    Plan handling class.

    This class provides methods to handle the plan configuration.

    Parameters
    ----------
    modality : {'photon', 'proton'}
        Treatment modality, needs to be consistent with the dose calculation \
        inputs.

    components : dict
        Optimization components for each segment of interest, i.e., \
        objective functions and constraints.

    Attributes
    ----------
    modality : {'photon', 'proton'}
        See 'Parameters'.

    components : dict
        See 'Parameters'.
    """

    def __init__(
            self,
            modality,
            components):

        # Log a message about the initialization of the class
        Datahub().logger.display_info("Initializing plan handler ...")

        # Get the input attributes
        self.modality = modality
        self.components = components

    def generate(self):
        """Generate the plan configuration dictionary."""

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the plan generation
        hub.logger.display_info(
            f"Generating plan configuration for {self.modality} treatment ...")

        # Initialize the plan dictionary
        plan_configuration = {
            'modality': self.modality,
            'RBE': 1.0 + 0.1*(self.modality == 'proton')}

        # Enter the plan dictionary into the datahub
        hub.plan_configuration = plan_configuration

        # Set the optimization components
        self.set_optimization_components()

    def set_optimization_components(
            self,
            verbose=True):
        """
        Set the components of the optimization problem.

        Parameters
        ----------
        verbose : bool
            Indicator for logging output messages.
        """

        def set_component(component, segment, category, base_dict):
            """Set the component by its segment and type assignment."""

            # Check if verbose is True
            if verbose:

                # Log a message about setting the component
                logger.display_info(
                    f"Setting {category} '{component.name}' for "
                    f"{[segment]+component.link} ...")

            # Check if the component is already included in the base dictionary
            if component.track_id not in base_dict:

                # Add the instance to the base dictionary
                base_dict[component.track_id] = {
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

        # Get the logger and the segmentation data
        logger, segmentation = hub.logger, hub.segmentation

        # Loop over the segments
        for segment in segmentation:

            # Reset the segment objective and constraint key
            segmentation[segment]['objective'] = None
            segmentation[segment]['constraint'] = None

        # Check if verbose is True
        if verbose:

            # Log a message about the components setting
            logger.display_info("Setting objectives and constraints ...")

        # Initialize the objective and constraint dictionaries
        objectives, constraints = {}, {}

        # Set the base dictionaries for the component types
        bases = {'objective': objectives, 'constraint': constraints}

        # Loop over the segments in the components dictionary
        for component in self.components:

            # Get the segment and component type
            segment, category = component.segment, component.component_type

            # Get the base dictionary
            base_dict = bases[category]

            # Set the component
            set_component(component, segment, category, base_dict)

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
