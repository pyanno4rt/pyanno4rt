"""Plan generation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Class definition


class PlanGenerator():
    """
    Plan generation class.

    This class provides methods to generate the plan configuration dictionary \
    for the management and retrieval of plan properties and plan-related \
    parameters.

    Parameters
    ----------
    modality : {'photon', 'proton'}
        Treatment modality, needs to be consistent with the dose calculation \
        inputs;
    """

    def __init__(
            self,
            modality):

        # Log a message about the initialization of the class
        Datahub().logger.display_info("Initializing plan generator ...")

        # Get the modality from the argument
        self.modality = modality

    def generate(self):
        """Generate the plan configuration dictionary."""

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the plan generation
        hub.logger.display_info("Generating plan configuration for "
                                f"{self.modality} treatment ...")

        # Initialize the plan dictionary
        plan_configuration = {'modality': self.modality}

        # Check if the modality is photons
        if self.modality == 'photon':

            # Add a neutral RBE to the dictionary
            plan_configuration['RBE'] = 1.0

        # Else, check if the modality is protons
        elif self.modality == 'proton':

            # Add a constant RBE to the dictionary
            plan_configuration['RBE'] = 1.1

        # Enter the plan dictionary into the datahub
        hub.plan_configuration = plan_configuration
