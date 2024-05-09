"""MATLAB data reading."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from scipy.io import loadmat

# %% Function definition


def read_data_from_mat(path):
    """
    Read the MATLAB data from the path.

    Parameters
    ----------
    path : str
        Path to the MATLAB file.

    Returns
    -------
    dict
        Dictionary with information on the CT slices.

    ndarray
        Array with information on the segmented structures.
    """

    # Load the MATLAB file
    data = loadmat(path, simplify_cells=True)

    return data['ct'], data['cst']
