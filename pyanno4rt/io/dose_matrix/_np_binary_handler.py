"""NumPy binary file handler."""

# Author: Tim Ortkamp

# %% External package import

from numpy import load
from scipy.sparse import csr_matrix

# %% Class definition


class NpBinHandler():
    """
    NumPy binary file handler class.

    This class provides methods to handle NumPy binary file-based \
    dose-influence matrices.
    """

    def __init__(self):

        pass

    def load(
            self,
            path,
            *args):
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

        return csr_matrix(load(path))

    def save(
            self,
            dose_matrix,
            path):
        """Save the dose-influence matrix."""
