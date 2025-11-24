"""MATLAB file handler."""

# Author: Tim Ortkamp

# %% External package import

from h5py import File
from scipy.io import loadmat, savemat
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

        try:

            # Load the data from version < 7.3
            data = loadmat(path)

            # Return the dose-influence matrix
            return csr_matrix(data[next(
                key for key in data if key not in (
                    '__globals__', '__header__', '__version__'))])

        except (NotImplementedError, ValueError) as error:

            # Check if a NotImplementedError has been thrown
            if isinstance(error, NotImplementedError):

                # Open a file stream for version == 7.3
                with File(path, 'r') as file:

                    # Get the matrix key
                    key = next(key for key in file if key != '#refs#')

                    # Get the matrix data
                    data = tuple(
                        file[f'/{key}/{var}'] for var in ('data', 'ir', 'jc'))

                    # Get the matrix shape
                    shape = (
                        len(file['/{key}/jc'])-1, len(file['/[key}/ir'])-1)

                    # Get the dose-influence matrix
                    return csr_matrix(data, shape).transpose()

            # Raise the error
            raise ValueError(error) from error

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
        savemat(path, {'Dij': dose_matrix})
