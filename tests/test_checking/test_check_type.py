"""Type check function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.checking import check_type

# %% Test definition


# Define the supported argument sets
@mark.parametrize(
    'label, data, options, type_condition',
    [('label', 'A', str, None),
     ('label',  'A', {'supported': str, 'unsupported':  int}, 'supported')],
    ids=['no type condition', 'type condition'])
def test_check_type_positive(label, data, options, type_condition):
    """Test the 'check_type' function with supported input."""

    # Assert the run-through of the function
    assert check_type(label, data, options, type_condition) is None


# Define the unsupported argument sets
@mark.parametrize(
    'label, data, options, type_condition',
    [('label', 'A', int, None),
     ('label', 'A', {'supported': str, 'unsupported':  int}, 'unsupported')],
    ids=['no type condition', 'type condition'])
def test_check_type_negative(label, data, options, type_condition):
    """Test the 'check_type' function with unsupported input."""

    # Assert the raise of an exception
    with raises(TypeError):
        check_type(label, data, options, type_condition)
