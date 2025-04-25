"""Vector length check function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.checking import check_length

# %% Test definition


# Define the supported argument sets
@mark.parametrize(
    'label, data, reference, sign',
    [('label', [1, 2, 3], 3, '=='), ('label', [1, 2, 3], 2, '>'),
     ('label', [1, 2, 3], 3, '>='), ('label', [1, 2, 3], 4, '<'),
     ('label', [1, 2, 3], 3, '<=')],
    ids=['equal', 'greater than', 'greater than or equal', 'less than',
         'less than or equal'])
def test_check_length_positive(label, data, reference, sign):
    """Test the 'check_length' function with supported input."""

    # Assert the run-through of the function
    assert check_length(label, data, reference, sign) is None


# Define the unsupported argument sets
@mark.parametrize(
    'label, data, reference, sign',
    [('label', [1, 2, 3], 2, '=='), ('label', [1, 2, 3], 4, '>'),
     ('label', [1, 2, 3], 4, '>='), ('label', [1, 2, 3], 2, '<'),
     ('label', [1, 2, 3], 2, '<=')],
    ids=['equal', 'greater than', 'greater than or equal', 'less than',
         'less than or equal'])
def test_check_length_negative(label, data, reference, sign):
    """Test the 'check_length' function with unsupported input."""

    # Assert the raise of an exception
    with raises(ValueError):
        check_length(label, data, reference, sign)
