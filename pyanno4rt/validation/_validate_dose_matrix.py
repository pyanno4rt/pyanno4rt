"""Dose-influence matrix validation."""

# Author: Tim Ortkamp

# %% External package import

from numpy import prod

# %% Function definition


def validate_dose_matrix(dose_shape, dose_matrix):
    """
    Validation function for the dose-influence matrix.

    Parameters
    ----------
    dose_shape : tuple
        Tuple with the number of dose grid points per axis.

    dose_matrix : object of class :class:`~scipy.sparse.csr_matrix`
        The object used to represent the dose-influence matrix.

    Raises
    ------
    ValueError
        If the number of voxels implied by the dose shape differs from the \
        number of dose-influence matrix rows, or if any element in the \
        dose-influence matrix is negative.
    """

    # Get the number of voxels from the dose matrix
    number_of_voxels = dose_matrix.shape[0]

    # Check if the calculated and implied number of dose voxels differs
    if prod(dose_shape) != number_of_voxels:

        # Raise an error
        raise ValueError(
            "The dose grid resolution from the treatment plan implies "
            f"{prod(dose_shape)} voxels, but the length of the first "
            f"dimension of the dose-influence matrix is {number_of_voxels}!")

    # Check if the dose-influence matrix has negative values
    if dose_matrix.min() < 0:

        # Raise an error
        raise ValueError(
            "The dose-influence matrix should only have non-negative entries, "
            "but its minimum is below zero!")
