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
        inputs.

    Attributes
    ----------
    modality : {'photon', 'proton'}
        See 'Parameters'.
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
        hub.logger.display_info(
            f"Generating plan configuration for {self.modality} treatment ...")

        # Initialize the plan dictionary
        plan_configuration = {'modality': self.modality,
                              'RBE': 1.0 + 0.1*(self.modality == 'proton')}

        # Enter the plan dictionary into the datahub
        hub.plan_configuration = plan_configuration
