"""Subtype validation function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from pytest import raises

# %% Internal package import

from pyanno4rt.validation import validate_subtype

# %% Test definition


def test_validate_subtype_positive():
    """Test the 'validate_subtype' function with supported input."""

    # Assert the run-through of the function
    assert validate_subtype('label', ('A', 'B'), str) is None


def test_validate_subtype_negative():
    """Test the 'validate_subtype' function with unsupported input."""

    # Assert the raise of an exception
    with raises(TypeError):
        validate_subtype('label', ('A', 'B', 3), str)
