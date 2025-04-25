"""Subtype check function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from pytest import raises

# %% Internal package import

from pyanno4rt.checking import check_subtype

# %% Test definition


def test_check_subtype_positive():
    """Test the 'check_subtype' function with supported input."""

    # Assert the run-through of the function
    assert check_subtype('label', ('A', 'B'), str) is None


def test_check_subtype_negative():
    """Test the 'check_subtype' function with unsupported input."""

    # Assert the raise of an exception
    with raises(TypeError):
        check_subtype('label', ('A', 'B', 3), str)
