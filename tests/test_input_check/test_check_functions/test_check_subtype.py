"""Subtype check function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pytest import raises

# %% Internal package import

from pyanno4rt.input_check.check_functions import check_subtype

# %% Test definition


def test_check_subtype_valid():
    """Test the 'check_subtype' function with valid input."""

    # Assert the run-through of the function
    assert check_subtype('label', ('A', 'B'), str) is None


def test_check_subtype_invalid():
    """Test the 'check_subtype' function with invalid input."""

    # Assert the raise of a TypeError exception
    with raises(TypeError):
        check_subtype('label', ('A', 'B', 3), str)
