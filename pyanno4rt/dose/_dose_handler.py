"""Dose handling."""

# Author: Tim Ortkamp

# %% External package import

from os.path import splitext

from numpy import prod

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.io.dose_matrix import (
    MatHandler, NpBinHandler, SpSparseBinHandler)
from pyanno4rt.logging import get_logger
from pyanno4rt.tools import arange_with_endpoint
from pyanno4rt.validation import validate_dose_matrix

# %% Class definition


class DoseHandler():
    """
    Dose handling class.

    This class provides methods to handle dose data.

    Parameters
    ----------
    dose_resolution : list
        Size of the dose grid in [`mm`] per dimension.

    number_of_fractions : int
        Number of fractions according to the treatment scheme.

    Attributes
    ----------
    sources : dict
        Dictionary with information on the external file sources and handlers.

    dose_information : dict
        Dictionary with information on the dose.
    """

    # Map the path extensions to the handlers
    sources = {
        '.mat': ('MATLAB file', MatHandler),
        '.npy': ('NumPy binary file', NpBinHandler),
        '.npz': ('SciPy sparse binary file', SpSparseBinHandler)}

    def __init__(
            self,
            dose_resolution,
            number_of_fractions):

        # Log a message about the initialization of the class
        get_logger().info("Initializing dose handler ...")

        # Initialize the dose information dictionary
        self.dose_information = (
            {'resolution': dict(zip(('x', 'y', 'z'), dose_resolution)),
             'number_of_fractions': number_of_fractions})

    def compute_dij(self):
        """."""

    def load_dij(
            self,
            path):
        """
        Load the dose-influence matrix.

        Parameters
        ----------
        path : str
            Path to the dose-influence matrix.

        Returns
        -------
        csr_matrix
            Dose-influence matrix.
        """

        # Get the file string and handler
        source, handler = self.sources[splitext(path)[1]]

        # Log a message about the dose-influence matrix loading
        get_logger().info("Loading dose-influence matrix from %s ...", source)

        # Load the dose-influence matrix into the dictionary
        self.dose_information['dose_influence_matrix'] = handler().load(path)

    def save_dij(
            self,
            path):
        """
        Save the dose-influence matrix.

        Parameters
        ----------
        path : str
            Path for storing the dose-influence matrix.
        """

        # Get the file string and handler
        source, handler = self.sources[splitext(path)[1]]

        # Log a message about the dose-influence matrix saving
        get_logger().info("Saving dose-influence matrix to %s ...", source)

        # Save the dose-influence matrix
        handler().save(self.dose_information['dose_influence_matrix'], path)

    def generate(
            self,
            computed_tomography,
            plan_configuration,
            dose_matrix_path):
        """
        Generate the dose information.

        Parameters
        ----------
        computed_tomography : dict
            Dictionary with information on the CT images.

        plan_configuration : dict
            Dictionary with information on the plan.

        dose_matrix_path : str
            Path to the dose-influence matrix.
        """

        # Log a message about the dose information generation
        get_logger().info(
            "Generating dose information for %s treatment ...",
            plan_configuration['modality'])

        # Add the grid points for all dimensions
        self.dose_information |= {
            dimension: arange_with_endpoint(
                computed_tomography[dimension][0],
                computed_tomography[dimension][-1],
                self.dose_information['resolution'][dimension])
            for dimension in ('x', 'y', 'z')}

        # Add the dose cube dimensions
        self.dose_information['cube_dimensions'] = tuple(
            len(self.dose_information[dimension])
            for dimension in ('y', 'x', 'z'))

        # Add the total number of dose voxels
        self.dose_information['number_of_voxels'] = prod(
            self.dose_information['cube_dimensions'])

        # Add the dose-influence matrix
        self.load_dij(dose_matrix_path)

        # Validate the dose-influence matrix
        validate_dose_matrix(
            self.dose_information['cube_dimensions'],
            self.dose_information['dose_influence_matrix'])

        # Add the degrees of freedom (number of decision variables)
        self.dose_information['degrees_of_freedom'] = self.dose_information[
            'dose_influence_matrix'].shape[1]

        # Store the dose information
        Datahub().dose_information = self.dose_information
