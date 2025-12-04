"""Item-in-set validation function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.validation import validate_item_in_set

# %% Test definition


# Define the valid argument sets
@mark.parametrize(
    'label, item, options, condition',
    [
     ('label', 'A', ('A', 1), None),
     ('label', ['A'], ('A', 1), None),
     ('label', 'A', {'valid': ('A', 1), 'invalid': (1, 2)}, 'valid'),
     ('label', ['A'], {'valid': ('A', 1), 'invalid': (1, 2)}, 'valid')
     ],
    ids=[
        'no condition, string data',
        'no condition, list data',
        'condition, string data',
        'condition, list data'
        ]
    )
def test_validate_item_in_set_positive(label, item, options, condition):
    """Test the 'validate_item_in_set' function with valid input."""

    # Assert the run-through of the function
    assert validate_item_in_set(label, item, options, condition) is None


# Define the invalid argument sets
@mark.parametrize(
    'label, item, options, condition',
    [
     ('label', 'A', (1, 2), None),
     ('label', ['A'], (1, 2), None),
     ('label', 'A', {'valid': ('A', 1), 'invalid': (1, 2)}, 'invalid'),
     ('label', ['A'], {'valid': ('A', 1), 'invalid': (1, 2)}, 'invalid')
     ],
    ids=[
        'no condition, string data',
        'no condition, list data',
        'condition, string data',
        'condition, list data'
        ]
    )
def test_validate_item_in_set_negative(label, item, options, condition):
    """Test the 'validate_item_in_set' function with invalid input."""

    # Assert the raise of an exception
    with raises(ValueError):
        validate_item_in_set(label, item, options, condition)
