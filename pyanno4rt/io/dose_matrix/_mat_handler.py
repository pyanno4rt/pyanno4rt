"""MATLAB file handler."""

# Author: Tim Ortkamp

# %% External package import

from h5py import File
from scipy.io import loadmat
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
            path,
            *args):
        """
        Load the dose-influence matrix.

        Parameters
        ----------
        path : str
            Path to the dose-influence matrix.

        number_of_voxels : int
            Number of fractions according to the treatment scheme.

        Returns
        -------
        csr_matrix
            Dose-influence matrix.
        """

        try:

            # Load the data from version < 7.3
            data = loadmat(path)

            # Get the dose-influence matrix
            dose_matrix = csr_matrix(data[next(
                key for key in data if key not in (
                    '__globals__', '__header__', '__version__'))])

        except (NotImplementedError, ValueError) as error:

            # Check if the error is NotImplementedError
            if isinstance(error, NotImplementedError):

                # Open a file stream for version == 7.3
                with File(path, 'r') as file:

                    # Get the matrix key
                    key = next(key for key in file if key != '#refs#')

                    # Get the dose-influence matrix
                    dose_matrix = csr_matrix(
                        (file[f'/{key}/data'], file[f'/{key}/ir'],
                         file[f'/{key}/jc']), shape=(
                            len(file['/D/jc'])-1, args[0])
                        ).transpose()

            else:

                # Raise the error
                raise ValueError(error) from error

        return dose_matrix

    def save(
            self,
            dose_matrix,
            path):
        """Save the dose-influence matrix."""
