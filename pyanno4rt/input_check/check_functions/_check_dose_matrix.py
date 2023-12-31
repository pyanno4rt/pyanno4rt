"""Dose-influence matrix checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import prod

# %% Function definition


def check_dose_matrix(cube_dimensions, matrix_dimension):
    """
    Check the equality of the cube dimensions product and the first dimension \
    of the dose-influence matrix.

    Parameters
    ----------
    cube_dimensions : tuple
        Tuple with the dimensions of the dose cube as calculated from the \
        dose resolution inputs.

    matrix_dimension : int
        Number of elements of the dose-influence matrix in the first dimension.

    Raises
    ------
    ValueError
        Raise when the cube dimensions product and the matrix dimension are \
        unequal.
    """
    # Check if the cube dimensions product and the matrix dimension are unequal
    if prod(cube_dimensions) != matrix_dimension:

        # Raise an error to indicate a wrong value for the cube dimensions
        raise ValueError(
            "The dose grid resolution passed in the treatment plan implies {} "
            "voxels, but the length of the first dimension of the "
            "dose-influence matrix is {}!"
            .format(prod(cube_dimensions),
                    matrix_dimension))
