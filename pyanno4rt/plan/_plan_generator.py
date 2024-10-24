"""Plan generation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from functools import reduce
from numpy import (
    ravel_multi_index, setdiff1d, union1d, unravel_index, where, zeros)
from scipy.ndimage import zoom

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.components import component_map
from pyanno4rt.tools import apply, flatten

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

    def preprocess(self):
        """Preprocess the optimization components and segmentations."""

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the components preprocessing
        hub.logger.display_info(
            "Preprocessing components and segmentations for optimization ...")

        # Set the objective and constraint functions
        objectives, constraints = PlanGenerator.set_optimization_components(
            self.components)

        # Remove overlaps between segments according to their priority
        PlanGenerator.remove_overlap(objectives | constraints)

        # Resize the segments to the dose grid
        PlanGenerator.resize_segments_to_dose()

        # Adjust the dose-volume-related parameters for fractionation
        PlanGenerator.adjust_parameters_for_fractionation(
            objectives | constraints)

        # Enter the objectives and constraints into the datahub
        hub.plan_configuration['objectives'] = objectives
        hub.plan_configuration['constraints'] = constraints

    @staticmethod
    def set_optimization_components(components):
        """
        Set the components of the optimization problem.

        Parameters
        ----------
        components : dict
            Optimization components for each segment of interest, i.e., \
            objectives and constraints, in the raw user format.

        Returns
        -------
        dict
            Dictionary with the internally configured objectives.

        dict
            Dictionary with the internally configured constraints.
        """

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
                **component['parameters'])

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
        for segment in components:

            # Check if the segment holds a list of components
            if isinstance(components[segment], list):

                # Loop over the component list
                for element in components[segment]:

                    # Get the category and component
                    category, component = element.values()

                    # Get the base dictionary
                    base_dict = bases[category]

                    # Set the component
                    set_component(component, segment, category, base_dict)

            else:

                # Get the category and component
                category, component = components[segment].values()

                # Get the base dictionary
                base_dict = bases[category]

                # Set the component
                set_component(component, segment, category, base_dict)

        return objectives, constraints

    @staticmethod
    def remove_overlap(components):
        """
        Remove overlaps between segments.

        Parameters
        ----------
        components : dict
            Optimization components for each segment of interest, i.e., \
            objectives and constraints, in the raw user format.
        """

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the overlap removal
        hub.logger.display_info("Removing segment overlaps ...")

        # Get the segmentation data
        segmentation = hub.segmentation

        def remove_segment_overlap(reference):
            """Remove the overlap from a reference segment."""

            # Get the superior indices from all VOIs
            superior_indices = [
                segmentation[segment]['raw_indices']
                for segment in set(flatten(
                        [component['segments']
                         for component in components.values()]))
                if (segmentation[segment]['parameters']['priority']
                    < segmentation[reference]['parameters']['priority'])]

            # Enter the overlap-free (prioritized) indices into the datahub
            segmentation[reference]['prioritized_indices'] = setdiff1d(
                segmentation[reference]['raw_indices'],
                reduce(union1d, superior_indices, -1))

        # Remove the overlaps from all segments
        apply(remove_segment_overlap, (*segmentation,))

    @staticmethod
    def resize_segments_to_dose():
        """Resize the segments from CT to dose grid."""

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the segment resizing
        hub.logger.display_info("Resizing segments from CT to dose grid ...")

        # Get the segmentation data
        segmentation = hub.segmentation

        # Get the CT and dose cube dimensions
        ct_dim, dose_dim = (hub.computed_tomography['cube_dimensions'],
                            hub.dose_information['cube_dimensions'])

        def resize_segment(segment):
            """Resize a segment to the dose grid."""

            # Initialize the segment mask
            mask = zeros(ct_dim)

            # Fill the mask at the indices of the segment
            mask[unravel_index(
                segmentation[segment]['prioritized_indices'], ct_dim,
                order='F')] = 1

            # Get the zoom factors for all cube dimensions
            zooms = (pair[0]/pair[1] for pair in zip(dose_dim, ct_dim))

            # Enter the dose grid level (resized) indices into the datahub
            segmentation[segment]['resized_indices'] = ravel_multi_index(
                where(zoom(mask, zooms, order=0)), dose_dim, order='F')

        # Resize all segments
        apply(resize_segment, (*segmentation,))

    @staticmethod
    def adjust_parameters_for_fractionation(components):
        """
        Adjust the dose parameters according to the number of fractions.

        Parameters
        ----------
        components : dict
            Dictionary with the internally configured objectives/constraints.
        """

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the parameter adjustment
        hub.logger.display_info(
            "Adjusting dose parameters for fractionation ...")

        # Get the number of fractions
        number_of_fractions = hub.dose_information['number_of_fractions']

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

        # Adjust all non-adjusted components with dose-related parameters
        apply(adjust_component, (
            component['instance'] for component in components.values()
            if not component['instance'].adjusted_parameters
            and 'dose' in component['instance'].parameter_category))
