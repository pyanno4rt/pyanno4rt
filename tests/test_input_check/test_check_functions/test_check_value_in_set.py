"""Value set check function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.input_check.check_functions import check_value_in_set

# %% Test definition


# Define the valid argument sets
@mark.parametrize(
    'label, data, options, value_condition',
    [('label', 'A', ('A', 1), None), ('label', ['A'], ('A', 1), None),
     ('label', 'A', {'valid': ('A', 1), 'invalid': (1, 2)}, 'valid'),
     ('label', ['A'], {'valid': ('A', 1), 'invalid': (1, 2)}, 'valid')],
    ids=['no value_condition, string data', 'no value_condition, list data',
         'value_condition, string data', 'value_condition, list data'])
def test_check_value_in_set_valid(label, data, options, value_condition):
    """Test the 'check_value_in_set' function with valid input."""

    # Assert the run-through of the function
    assert check_value_in_set(label, data, options, value_condition) is None


# Define the invalid argument sets
@mark.parametrize(
    'label, data, options, value_condition',
    [('label', 'A', (1, 2), None), ('label', ['A'], (1, 2), None),
     ('label', 'A', {'valid': ('A', 1), 'invalid': (1, 2)}, 'invalid'),
     ('label', ['A'], {'valid': ('A', 1), 'invalid': (1, 2)}, 'invalid')],
    ids=['no value_condition, string data', 'no value_condition, list data',
         'value_condition, string data', 'value_condition, list data'])
def test_check_value_in_set_error(label, data, options, value_condition):
    """Test the 'check_value_in_set' function with invalid input."""

    # Assert the raise of a ValueError exception
    with raises(ValueError):
        check_value_in_set(label, data, options, value_condition)
