"""Plan generation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.components import component_map

# %% Class definition


class PlanGenerator():
    """
    Plan generation class.

    This class provides methods to generate the plan configuration dictionary \
    for the management and retrieval of plan properties and plan-related \
    parameters, including the plan objectives and constraints.

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
        Datahub().logger.display_info("Initializing plan generator ...")

        # Get the attributes from the arguments
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

    def set_optimization_components(self):
        """Set the components of the optimization problem."""

        # Initialize the datahub
        hub = Datahub()

        # Get the logger and the segmentation data
        logger, segmentation = hub.logger, hub.segmentation

        # Loop over the segments
        for segment in segmentation:

            # Reset the segment objective and constraint key
            segmentation[segment]['objective'] = None
            segmentation[segment]['constraint'] = None

        # Log a message about the components setting
        logger.display_info("Setting objectives and constraints ...")

        # Initialize the objective and constraint dictionaries
        objectives, constraints = {}, {}

        # Set the base dictionaries for the component types
        bases = {'objective': objectives, 'constraint': constraints}

        def set_component(component, segment, category, base_dict):
            """Set the component by its segment and type assignment."""

            # Get the instance from the component map
            instance = component_map[component['class']](
                segment=segment, **component['parameters'])

            # Log a message about setting the instance
            logger.display_info(
                f"Setting {category} '{instance.name}' for "
                f"{[segment]+instance.link} ...")

            # Get the instance key for the base dictionary
            instance_key = '-'.join(filter(
                None, (f"{[segment]+instance.link}", instance.name,
                       instance.identifier)))

            # Check if the instance is already included in the base dictionary
            if instance_key not in base_dict:

                # Add the instance to the base dictionary
                base_dict[instance_key] = {
                    'segments': [segment]+instance.link,
                    'instance': instance}

                # Check if no instance has been set yet
                if not segmentation[segment][category]:

                    # Add the instance to the segment
                    segmentation[segment][category] = instance

                else:

                    # Check if the component is a list
                    if isinstance(segmentation[segment][category], list):

                        # Append the instance
                        segmentation[segment][category].append(instance)

                    else:

                        # Make a list and add the instance
                        segmentation[segment][category] = [
                            segmentation[segment][category], instance]

        # Loop over the segments in the components dictionary
        for segment in self.components:

            # Check if the segment holds a list of components
            if isinstance(self.components[segment], list):

                # Loop over the component list
                for element in self.components[segment]:

                    # Get the category and component
                    category, component = element.values()

                    # Get the base dictionary
                    base_dict = bases[category]

                    # Set the component
                    set_component(component, segment, category, base_dict)

            else:

                # Get the category and component
                category, component = self.components[segment].values()

                # Get the base dictionary
                base_dict = bases[category]

                # Set the component
                set_component(component, segment, category, base_dict)

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
