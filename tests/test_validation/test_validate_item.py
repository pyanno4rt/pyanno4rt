"""Item validation function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.validation import validate_item

# %% Test definition


# Define the valid argument sets
@mark.parametrize(
    'label, item, reference, sign',
    [
     ('label', None, 1, '=='),
     ('label', 1, 1, '=='),
     ('label', (1, 2, 3, 4), 5, '<')
     ],
    ids=[
        'no data',
        'scalar',
        'vector'
        ]
    )
def test_validate_item_positive(label, item, reference, sign):
    """Test the 'validate_item' function with valid input."""

    # Assert the run-through of the function
    assert validate_item(label, item, reference, sign) is None


# Define the invalid argument sets
@mark.parametrize(
    'label, item, reference, sign',
    [
     ('label', 1, 2, '>='),
     ('label', (1, 2, 3, 4), 5, '==')
     ],
    ids=[
        'scalar',
        'vector'
        ]
    )
def test_validate_item_negative(label, item, reference, sign):
    """Test the 'validate_item' function with invalid input."""

    # Assert the raise of an exception
    with raises(ValueError):
        validate_item(label, item, reference, sign)
