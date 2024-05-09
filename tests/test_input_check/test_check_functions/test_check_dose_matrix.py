"""Dose-influence matrix check function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pytest import raises

# %% Internal package import

from pyanno4rt.input_check.check_functions import check_dose_matrix

# %% Test definition


def test_check_dose_matrix_valid():
    """Test the 'check_dose_matrix' function with valid input."""

    # Assert the run-through of the function
    assert check_dose_matrix((1, 2, 3), 6) is None


def test_check_dose_matrix_invalid():
    """Test the 'check_dose_matrix' function with invalid input."""

    # Assert the raise of a ValueError exception
    with raises(ValueError):
        check_dose_matrix((1, 2, 3), 5)
