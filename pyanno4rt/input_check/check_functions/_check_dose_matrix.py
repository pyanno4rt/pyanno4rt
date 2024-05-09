"""Dose-influence matrix checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import prod

# %% Function definition


def check_dose_matrix(dose_shape, dose_matrix_rows):
    """
    Check the equality between the number of dose voxels calculated from the \
    dose resolution inputs and implied by the dose-influence matrix.

    Parameters
    ----------
    dose_shape : tuple
        Tuple with the number of dose grid points per axis, calculated from \
        the dose resolution inputs.

    dose_matrix_rows : int
        Number of rows in the dose-influence matrix (the number of voxels in \
        the dose grid).

    Raises
    ------
    ValueError
        If the product of the elements in dose_shape is not equal to the \
        value of dose_matrix_rows.
    """

    # Check if the calculated and implied number of dose voxels differs
    if prod(dose_shape) != dose_matrix_rows:

        # Raise an error to indicate a difference in the number of dose voxels
        raise ValueError(
            "The dose grid resolution passed in the treatment plan implies "
            f"{prod(dose_shape)} voxels, but the length of the first "
            f"dimension of the dose-influence matrix is {dose_matrix_rows}!")
