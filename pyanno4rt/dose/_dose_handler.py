"""Dose handling."""

# Author: Tim Ortkamp

# %% External package import

from os.path import splitext

from numpy import prod

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.io.dose_matrix import (
    MatHandler, NpBinHandler, SpSparseBinHandler)
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
    dose_resolution : list
        See 'Parameters'.

    number_of_fractions : int
        See 'Parameters'.
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
        Datahub().logger.display_info("Initializing dose handler ...")

        # Get the input attributes
        self.dose_resolution = dose_resolution
        self.number_of_fractions = number_of_fractions

        # Initialize the dose information dictionary
        self.dose_information = {}

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

        # Initialize the datahub
        hub = Datahub()

        # Get the file string and handler
        source, handler = self.sources[splitext(path)[1]]

        # Log a message about the dose-influence matrix loading
        hub.logger.display_info(
            f"Loading dose-influence matrix from {source} ...")

        return handler().load(path, self.number_of_fractions)

    def save_dij(
            self,
            path):
        """
        Save the patient imaging data.

        Parameters
        ----------
        path : str
            Path for storing the patient imaging data.
        """

        # Initialize the datahub
        hub = Datahub()

        # Get the file string and handler
        source, handler = self.sources[splitext(path)[1]]

        # Log a message about the dose-influence matrix saving
        hub.logger.display_info(
            f"Saving dose-influence matrix from {source} ...")

        # Save the patient imaging data
        handler().save(self.dose_information['dose_influence_matrix'], path)

    def load(
            self,
            dose_matrix_path):
        """
        Load the dose data.

        Parameters
        ----------
        dose_matrix_path : str
            Path to the dose-influence matrix.
        """

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the dose information dictionary loading
        hub.logger.display_info(
            "Loading dose information dictionary for "
            f"{hub.plan_configuration['modality']} treatment ...")

        # Add the dose grid resolution
        self.dose_information['resolution'] = dict(zip(
            ('x', 'y', 'z'), self.dose_resolution))

        # Add the grid points for all dimensions
        self.dose_information |= {
            dimension: arange_with_endpoint(
                hub.computed_tomography[dimension][0],
                hub.computed_tomography[dimension][-1],
                self.dose_information['resolution'][dimension])
            for dimension in ('x', 'y', 'z')}

        # Add the dose cube dimensions
        self.dose_information['cube_dimensions'] = tuple(
            len(self.dose_information[dimension])
            for dimension in ('y', 'x', 'z'))

        # Add the total number of dose voxels
        self.dose_information['number_of_voxels'] = prod(
            self.dose_information['cube_dimensions'])

        # Add the number of fractions
        self.dose_information['number_of_fractions'] = self.number_of_fractions

        # Add the dose-influence matrix
        self.dose_information['dose_influence_matrix'] = self.load_dij(
            dose_matrix_path)

        # Validate the dose-influence matrix
        validate_dose_matrix(
            self.dose_information['cube_dimensions'],
            self.dose_information['dose_influence_matrix'])

        # Add the degrees of freedom (number of decision variables)
        self.dose_information['degrees_of_freedom'] = self.dose_information[
            'dose_influence_matrix'].shape[1]

        # Store the dose information
        hub.dose_information = self.dose_information
