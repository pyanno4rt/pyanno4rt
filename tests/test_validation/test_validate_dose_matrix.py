"""Dose-influence matrix validation function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from numpy import nan
from pytest import mark, raises
from scipy.sparse import csr_matrix

# %% Internal package import

from pyanno4rt.validation import validate_dose_matrix

# %% Test definition


# Define the valid argument sets
@mark.parametrize(
    'dose_shape, dose_matrix',
    [
     ((1, 2, 3), csr_matrix((6, 1))),
     ((3, 1), csr_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
     ],
    ids=[
        '3D dose, equal beamlet number',
        '2D dose, equal beamlet number'
        ]
    )
def test_validate_dose_matrix_positive(dose_shape, dose_matrix):
    """Test the 'validate_dose_matrix' function with valid input."""

    # Assert the run-through of the function
    assert validate_dose_matrix(dose_shape, dose_matrix) is None


# Define the invalid argument sets
@mark.parametrize(
    'dose_shape, dose_matrix',
    [
     ((1, 2, 3), csr_matrix((5, 1))),
     ((3, 1), csr_matrix([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])),
     ((3, 1), csr_matrix([[1, nan, 0], [0, 1, 0], [0, 0, 1]]))
     ],
    ids=[
        '3D dose, different beamlet number',
        '2D dose, negative values',
        '2D dose, NaN values'
        ]
    )
def test_validate_dose_matrix_negative(dose_shape, dose_matrix):
    """Test the 'validate_dose_matrix' function with invalid input."""

    # Assert the raise of an exception
    with raises(ValueError):
        validate_dose_matrix(dose_shape, dose_matrix)
