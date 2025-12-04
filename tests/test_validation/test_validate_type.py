"""Type validation function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.validation import validate_type

# %% Test definition


# Define the valid argument sets
@mark.parametrize(
    'label, item, options, condition',
    [
     ('label', 'A', str, None),
     ('label',  'A', {'valid': str, 'invalid':  int}, 'valid')
     ],
    ids=[
        'no condition',
        'condition'
        ]
    )
def test_validate_type_positive(label, item, options, condition):
    """Test the 'validate_type' function with valid input."""

    # Assert the run-through of the function
    assert validate_type(label, item, options, condition) is None


# Define the invalid argument sets
@mark.parametrize(
    'label, item, options, condition',
    [
     ('label', 'A', int, None),
     ('label', 'A', {'valid': str, 'invalid':  int}, 'invalid')
     ],
    ids=[
        'no condition',
        'condition'
        ]
    )
def test_validate_type_negative(label, item, options, condition):
    """Test the 'validate_type' function with invalid input."""

    # Assert the raise of an exception
    with raises(TypeError):
        validate_type(label, item, options, condition)
