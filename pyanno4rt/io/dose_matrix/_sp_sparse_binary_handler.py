"""SciPy sparse binary file handler."""

# Author: Tim Ortkamp

# %% External package import

from scipy.sparse import load_npz

# %% Class definition


class SpSparseBinHandler():
    """
    SciPy sparse binary file handler class.

    This class provides methods to handle SciPy sparse binary file-based \
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

        return load_npz(path)

    def save(
            self,
            dose_matrix,
            path):
        """Save the dose-influence matrix."""
