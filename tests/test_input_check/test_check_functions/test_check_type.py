"""Type check function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.input_check.check_functions import check_type

# %% Test definition


# Define the valid argument sets
@mark.parametrize(
    'label, data, types, type_condition',
    [('label', 'A', str, None),
     ('label',  'A', {'valid': str, 'invalid':  int}, 'valid')],
    ids=['no type condition', 'type condition'])
def test_check_type_valid(label, data, types, type_condition):
    """Test the 'check_type' function with valid input."""

    # Assert the run-through of the function
    assert check_type(label, data, types, type_condition) is None


# Define the invalid argument sets
@mark.parametrize(
    'label, data, types, type_condition',
    [('label', 'A', int, None),
     ('label', 'A', {'valid': str, 'invalid':  int}, 'invalid')],
    ids=['no type condition', 'type condition'])
def test_check_type_invalid(label, data, types, type_condition):
    """Test the 'check_type' function with invalid input."""

    # Assert the raise of a TypeError exception
    with raises(TypeError):
        check_type(label, data, types, type_condition)
