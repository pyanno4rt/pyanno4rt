"""Vector length validation function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.validation import validate_length

# %% Test definition


# Define the valid argument sets
@mark.parametrize(
    'label, item, reference, sign',
    [
     ('label', [1, 2, 3], 3, '=='),
     ('label', [1, 2, 3], 2, '>'),
     ('label', [1, 2, 3], 3, '>='),
     ('label', [1, 2, 3], 4, '<'),
     ('label', [1, 2, 3], 3, '<=')
     ],
    ids=[
        'equal',
        'greater than',
        'greater than or equal',
        'less than',
        'less than or equal'
        ]
    )
def test_validate_length_positive(label, item, reference, sign):
    """Test the 'validate_length' function with valid input."""

    # Assert the run-through of the function
    assert validate_length(label, item, reference, sign) is None


# Define the invalid argument sets
@mark.parametrize(
    'label, item, reference, sign',
    [
     ('label', [1, 2, 3], 2, '=='),
     ('label', [1, 2, 3], 4, '>'),
     ('label', [1, 2, 3], 4, '>='),
     ('label', [1, 2, 3], 2, '<'),
     ('label', [1, 2, 3], 2, '<=')
     ],
    ids=[
        'equal',
        'greater than',
        'greater than or equal',
        'less than',
        'less than or equal'
        ]
    )
def test_validate_length_negative(label, item, reference, sign):
    """Test the 'validate_length' function with invalid input."""

    # Assert the raise of an exception
    with raises(ValueError):
        validate_length(label, item, reference, sign)
