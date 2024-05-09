"""Dictionary key check function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.input_check.check_functions import check_key_in_dict

# %% Test definition


# Define the valid argument sets
@mark.parametrize(
    'label, data, keys',
    [('label', {'A': 1}, ('A',)), ('label', {'A': 1, 'B': 2}, ('A', 'B'))],
    ids=['single-key', 'multi-key'])
def test_check_key_in_dict_valid(label, data, keys):
    """Test the 'check_key_in_dict' function with valid input."""

    # Assert the run-through of the function
    assert check_key_in_dict(label, data, keys) is None


# Define the invalid argument sets
@mark.parametrize(
    'label, data, keys',
    [('label', {'A': 1}, ('B',)), ('label', {'A': 1, 'B': 2}, ('A', 'C'))],
    ids=['single-key', 'multi-key'])
def test_check_key_in_dict_invalid(label, data, keys):
    """Test the 'check_key_in_dict' function with invalid input."""

    # Assert the raise of a KeyError exception
    with raises(KeyError):
        check_key_in_dict(label, data, keys)
