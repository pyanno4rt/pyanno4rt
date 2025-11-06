"""Dose-influence matrix validation function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from pytest import raises
from scipy.sparse import csr_matrix

# %% Internal package import

from pyanno4rt.validation import validate_dose_matrix

# %% Test definition


def test_validate_dose_matrix_positive():
    """Test the 'validate_dose_matrix' function with supported input."""

    # Assert the run-through of the function
    assert validate_dose_matrix((1, 2, 3), csr_matrix((6, 1))) is None


def test_validate_dose_matrix_negative():
    """Test the 'validate_dose_matrix' function with unsupported input."""

    # Assert the raise of an exception
    with raises(ValueError):
        validate_dose_matrix((1, 2, 3), csr_matrix((5, 1)))
