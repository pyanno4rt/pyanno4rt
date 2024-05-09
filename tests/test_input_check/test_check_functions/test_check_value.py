"""Value check function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.input_check.check_functions import check_value

# %% Test definition


# Define the valid argument sets
@mark.parametrize(
    'label, data, reference, sign, is_vector',
    [('label', None, 1, '==', False), ('label', 1, 1, '==', False),
     ('label', (1, 2, 3, 4), 5, '<', True)],
    ids=['no data', 'scalar', 'vector'])
def test_check_value_valid(label, data, reference, sign, is_vector):
    """Test the 'check_value' function with valid input."""

    # Assert the run-through of the function
    assert check_value(label, data, reference, sign, is_vector) is None


# Define the invalid argument sets
@mark.parametrize(
    'label, data, reference, sign, is_vector',
    [('label', 1, 2, '>=', False), ('label', (1, 2, 3, 4), 5, '==', True)],
    ids=['scalar', 'vector'])
def test_check_value_invalid(label, data, reference, sign, is_vector):
    """Test the 'check_value' function with invalid input."""

    # Assert the raise of a ValueError exception
    with raises(ValueError):
        check_value(label, data, reference, sign, is_vector)
