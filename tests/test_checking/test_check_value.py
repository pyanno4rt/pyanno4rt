"""Value check function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.checking import check_value

# %% Test definition


# Define the supported argument sets
@mark.parametrize(
    'label, data, reference, sign',
    [('label', None, 1, '=='), ('label', 1, 1, '=='),
     ('label', (1, 2, 3, 4), 5, '<')],
    ids=['no data', 'scalar', 'vector'])
def test_check_value_positive(label, data, reference, sign):
    """Test the 'check_value' function with supported input."""

    # Assert the run-through of the function
    assert check_value(label, data, reference, sign) is None


# Define the unsupported argument sets
@mark.parametrize(
    'label, data, reference, sign',
    [('label', 1, 2, '>='), ('label', (1, 2, 3, 4), 5, '==')],
    ids=['scalar', 'vector'])
def test_check_value_negative(label, data, reference, sign):
    """Test the 'check_value' function with unsupported input."""

    # Assert the raise of an exception
    with raises(ValueError):
        check_value(label, data, reference, sign)
