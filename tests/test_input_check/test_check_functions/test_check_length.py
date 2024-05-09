"""Vector length check function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.input_check.check_functions import check_length

# %% Test definition


# Define the valid argument sets
@mark.parametrize(
    'label, data, reference, sign',
    [('label', [1, 2, 3], 3, '=='), ('label', [1, 2, 3], 2, '>'),
     ('label', [1, 2, 3], 3, '>='), ('label', [1, 2, 3], 4, '<'),
     ('label', [1, 2, 3], 3, '<=')],
    ids=['equal', 'greater than', 'greater than or equal', 'less than',
         'less than or equal'])
def test_check_length_valid(label, data, reference, sign):
    """Test the 'check_length' function with valid input."""

    # Assert the run-through of the function
    assert check_length(label, data, reference, sign) is None


# Define the invalid argument sets
@mark.parametrize(
    'label, data, reference, sign',
    [('label', [1, 2, 3], 2, '=='), ('label', [1, 2, 3], 4, '>'),
     ('label', [1, 2, 3], 4, '>='), ('label', [1, 2, 3], 2, '<'),
     ('label', [1, 2, 3], 2, '<=')],
    ids=['equal', 'greater than', 'greater than or equal', 'less than',
         'less than or equal'])
def test_check_length_invalid(label, data, reference, sign):
    """Test the 'check_length' function with invalid input."""

    # Assert the raise of a ValueError exception
    with raises(ValueError):
        check_length(label, data, reference, sign)
