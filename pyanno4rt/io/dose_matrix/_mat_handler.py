"""MATLAB file handler."""

# Author: Tim Ortkamp

# %% External package import

from pymatreader import read_mat
from scipy.io import savemat
from scipy.sparse import csr_matrix

# %% Class definition


class MatHandler():
    """
    MATLAB file handler class.

    This class provides methods to handle MATLAB file-based dose-influence \
    matrices.
    """

    def __init__(self):

        pass

    def load(
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

        # Load the data from version < 7.3
        data = read_mat(path, ['Dij'])

        # Return the dose-influence matrix
        return csr_matrix(data[next(
            key for key in data if key not in (
                '__globals__', '__header__', '__version__'))])

    def save(
            self,
            dose_matrix,
            path):
        """
        Save the dose-influence matrix.

        Parameters
        ----------
        dose_matrix : csr_matrix
            Dose-influence matrix.

        path : str
            Path for storing the dose-influence matrix.
        """

        # Save the matrix to the path
        savemat(path, {'Dij': dose_matrix}, do_compression=True)
