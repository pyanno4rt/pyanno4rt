"""Fluence initialization."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import hstack, ones

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import flatten

# %% Class definition


class FluenceInitializer():
    """
    Fluence initialization class.

    This class implements various strategies to initialize the fluence vector \
    for the optimizer, including the initialization from a given vector or \
    from a reference plan. It also features strategies towards coverage of \
    the target volumes and characterization of the model training data.

    Parameters
    ----------
    initial_strategy : string
        Initialization strategy for the fluence vector.

    initial_fluence_vector: ndarray
        Input vector for initializing the fluence vector by a warm start.

    Attributes
    ----------
    initial_strategy : string
        See 'Parameters'.

    initial_fluence_vector : ndarray
        See 'Parameters'.
    """

    def __init__(self,
                 initial_strategy,
                 initial_fluence_vector):

        # Log a message about the initialization of the fluence initializer
        Datahub().logger.display_info("Initializing fluence initializer ...")

        # Get the instance attributes from the arguments
        self.initial_strategy = initial_strategy
        self.initial_fluence_vector = initial_fluence_vector

    def run_strategy(self):
        """Run the initialization method based on the selected strategy."""
        # Map the 'initial_strategy' argument to the initialization methods
        strategies = {
            'target-coverage': FluenceInitializer.initialize_from_target,
            'warm-start': FluenceInitializer.initialize_from_vector}

        # Initialize the fluence vector based on the selected strategy
        initial_fluence = strategies[self.initial_strategy](
            self.initial_fluence_vector)

        return initial_fluence

    @staticmethod
    def initialize_from_target(_):
        """
        Initialize the fluence vector with respect to target coverage.

        Returns
        -------
        ndarray
            Initial fluence vector.
        """
        # Initialize the datahub
        hub = Datahub()

        # Log a message about the vector initialization
        hub.logger.display_info("Initializing fluence vector with respect to "
                                "target coverage ...")

        # Get the segmentation and dose information data from the datahub
        segmentation = hub.segmentation
        dose_information = hub.dose_information

        # Get the target segments
        segments = tuple(segment for segment in (*segmentation,)
                         if (segmentation[segment]['type'] == 'TARGET')
                         and (segmentation[segment]['objective']
                              or segmentation[segment]['constraint']))

        # Get the resized indices of the target segments
        indices = hstack([segmentation[segment]['resized_indices']
                          for segment in segments])

        def get_dose_parameters(segment):
            """Get the dose-related objective parameters of a segment."""
            # 
            values = []

            if isinstance(segmentation[segment]['objective'], (tuple, list)):

                for objective in segmentation[segment]['objective']:

                    # Get the positions of the dose-related parameters
                    positions = tuple(item[0] for item in enumerate(
                        objective.parameter_category)
                        if item[1] == 'dose')

                    # Check if more than one dose-related parameter is present
                    if len(positions) > 1:

                        # Return a tuple of dose-related parameters
                        values.extend([objective.parameter_value[pos]
                                       for pos in positions])

                    else:

                        # Otherwise, return the singular dose-related parameter
                        values.append([objective.parameter_value[pos]
                                       for pos in positions])

            else:

                # Get the positions of the dose-related parameters
                positions = tuple(item[0] for item in enumerate(
                    segmentation[segment]['objective'].parameter_category)
                    if item[1] == 'dose')

                # Check if more than one dose-related parameter is present
                if len(positions) > 1:

                    # Return a tuple of dose-related parameters
                    values.extend([segmentation[segment][
                        'objective'].parameter_value[pos]
                        for pos in positions])

                else:

                    # Otherwise, return the singular dose-related parameter
                    values.append(segmentation[
                        segment]['objective'].parameter_value[positions[0]])

            return values

        # Get the maximum dose parameter value
        all_values = []
        for segment in segments:
            all_values.append(get_dose_parameters(segment))

        # 
        max_dose = max(flatten(all_values))

        # Initialize a vector of ones
        ones_vector = ones((dose_information['degrees_of_freedom'],))

        return ones_vector * max_dose/(
            hub.plan_configuration['RBE'] * (
                dose_information['dose_influence_matrix'][indices, :]
                @ ones_vector).mean())

    @staticmethod
    def initialize_from_vector(initial_fluence_vector):
        """
        Initialize the fluence vector with respect to a reference optimal \
        point.

        Returns
        -------
        ndarray
            Initial fluence vector.
        """
        # Log a message about the vector initialization
        Datahub().logger.display_info("Initializing fluence vector with "
                                      "respect to a reference optimal point "
                                      "...")

        return initial_fluence_vector
