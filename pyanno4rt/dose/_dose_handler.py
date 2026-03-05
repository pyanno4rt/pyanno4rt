"""Dose handling."""

# Author: Tim Ortkamp

# %% External package import

from os.path import splitext

from numpy import float32, prod

# %% Internal package import

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
    sources : None or dict
        Dictionary with information on the external file sources and handlers.

    resolution : None or list
        Size of the dose grid in `[mm]` per dimension.

    number_of_fractions : None or int
        Number of fractions according to the treatment scheme.

    grid : None or dict
        Dictionary with the grid points in all dimensions.

    cube_dimensions : None or tuple
        Dose cube dimensions.

    number_of_voxels : None or int
        Number of dose voxels.

    dose_influence_matrix :  None or csr_matrix
        Dose-influence matrix.

    degrees_of_freedom : None or int
        Degrees of freedom (number of beamlets).
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

        # Get the instance attributes
        self.resolution = dict(zip(('x', 'y', 'z'), dose_resolution))
        self.number_of_fractions = number_of_fractions

        # Initialize the other attributes
        self.grid = None
        self.cube_dimensions = None
        self.number_of_voxels = None
        self.dose_influence_matrix = None
        self.degrees_of_freedom = None

    def compute_dij(self):
        """."""
        # A placeholder method for potential future dose calculation

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

        return handler().load(path)

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
        handler().save(self.dose_influence_matrix, path)

    def generate(
            self,
            computed_tomography,
            modality,
            dose_matrix_path):
        """
        Generate the dose information.

        Parameters
        ----------
        computed_tomography : dict
            Dictionary with information on the CT images.

        modality : {'photon', 'proton'}
            Treatment modality.

        dose_matrix_path : str
            Path to the dose-influence matrix.
        """

        # Log a message about the dose information generation
        get_logger().info(
            "Generating dose information for %s treatment ...", modality)

        # Get the grid points
        self.grid = {
            dimension: arange_with_endpoint(
                computed_tomography[dimension][0],
                computed_tomography[dimension][-1],
                self.resolution[dimension])
            for dimension in ('x', 'y', 'z')}

        # Get the cube dimensions
        self.cube_dimensions = tuple(
            len(self.grid[dimension]) for dimension in ('y', 'x', 'z'))

        # Get the total number of voxels
        self.number_of_voxels = prod(self.cube_dimensions)

        # Get the dose-influence matrix
        self.dose_influence_matrix = self.load_dij(dose_matrix_path)

        # Use single precision on the dose-influence matrix
        self.dose_influence_matrix = self.dose_influence_matrix.astype(float32)

        # Validate the dose-influence matrix
        validate_dose_matrix(self.cube_dimensions, self.dose_influence_matrix)

        # Get the degrees of freedom (number of beamlets)
        self.degrees_of_freedom = self.dose_influence_matrix.shape[1]
