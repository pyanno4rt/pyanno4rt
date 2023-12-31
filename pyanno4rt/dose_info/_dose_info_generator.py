"""Dose information generation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from h5py import File
from numpy import load, prod
from scipy.io import loadmat
from scipy.sparse import csr_matrix

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.input_check.check_functions import check_dose_matrix
from pyanno4rt.tools import arange_with_endpoint

# %% Class definition


class DoseInfoGenerator():
    """
    Dose information generation class.

    This class provides methods to generate the dose information dictionary \
    for the management and retrieval of dose grid properties and dose-related \
    parameters.

    Parameters
    ----------
    dose_resolution : list
        Size of the dose grid in [`mm`] per dimension, needs to be consistent \
        with the dose calculation inputs.

    number_of_fractions : int
        Number of fractions according to the treatment scheme.

    dose_path : string
        Path to the dose-influence matrix file (.mat or .npy).
    """

    def __init__(
            self,
            dose_resolution,
            number_of_fractions,
            dose_path):

        # Log a message about the initialization of the class
        Datahub().logger.display_info("Initializing dose information "
                                      "generator ...")

        # Get the instance attributes from the arguments
        self.dose_resolution = dose_resolution
        self.number_of_fractions = number_of_fractions
        self.dose_path = dose_path

    def generate(self):
        """Generate the dose information dictionary."""

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the dose information generation
        hub.logger.display_info("Generating dose information for "
                                f"{hub.plan_configuration['modality']} "
                                "treatment ...")

        # Initialize the dose information dictionary
        dose_information = {}

        # Add the dose grid resolution
        dose_information['resolution'] = dict(
            zip(('x', 'y', 'z'), self.dose_resolution))

        # Add the grid points for all dimensions
        dose_information.update({dimension: arange_with_endpoint(
            hub.computed_tomography[dimension][0],
            hub.computed_tomography[dimension][-1],
            dose_information['resolution'][dimension])
            for dimension in ('x', 'y', 'z')})

        # Add the dose cube dimensions
        dose_information['cube_dimensions'] = [
            len(dose_information[dimension]) for dimension in ('x', 'y', 'z')]

        # Add the total number of dose voxels
        dose_information['number_of_voxels'] = prod(
            dose_information['cube_dimensions'])

        # Add the number of fractions
        dose_information['number_of_fractions'] = self.number_of_fractions

        # Log a message about the dose-influence matrix addition
        hub.logger.display_info("Adding dose-influence matrix ...")

        # Check if the dose path leads to a .mat file
        if self.dose_path.endswith('.mat'):

            try:

                # Read the dose-influence matrix from version < 7.3
                dose_information['dose_influence_matrix'] = csr_matrix(
                    loadmat(self.dose_path)['Dij'][0][0])

            except NotImplementedError:

                # Open a file stream
                with File(self.dose_path, 'r') as file:

                    # Read the dose-influence matrix from version >= 7.3
                    dose_information['dose_influence_matrix'] = csr_matrix(
                        file['Dij'][0][0])

        # Else, check if the dose path leads to a .npy file
        elif self.dose_path.endswith('.npy'):

            # Add the dose-influence matrix from the .npy file
            dose_information['dose_influence_matrix'] = csr_matrix(
                load(self.dose_path))

        # Add the degrees of freedom (the number of decision variables)
        dose_information['degrees_of_freedom'] = dose_information[
            'dose_influence_matrix'].shape[1]

        # Check the dose resolution with the dose-influence matrix dimensions
        check_dose_matrix(dose_information['cube_dimensions'],
                          dose_information['dose_influence_matrix'].shape[0])

        # Enter the dose information dictionary into the datahub
        hub.dose_information = dose_information
