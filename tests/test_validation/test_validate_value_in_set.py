"""Value-in-set validation function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.validation import validate_value_in_set

# %% Test definition


# Define the supported argument sets
@mark.parametrize(
    'label, data, options, value_condition',
    [('label', 'A', ('A', 1), None), ('label', ['A'], ('A', 1), None),
     ('label', 'A', {'supp.': ('A', 1), 'unsupp.': (1, 2)}, 'supp.'),
     ('label', ['A'], {'supp.': ('A', 1), 'unsupp.': (1, 2)}, 'supp.')],
    ids=['no value_condition, string data', 'no value_condition, list data',
         'value_condition, string data', 'value_condition, list data'])
def test_validate_value_in_set_positive(label, data, options, value_condition):
    """Test the 'validate_value_in_set' function with supported input."""

    # Assert the run-through of the function
    assert validate_value_in_set(label, data, options, value_condition) is None


# Define the unsupported argument sets
@mark.parametrize(
    'label, data, options, value_condition',
    [('label', 'A', (1, 2), None), ('label', ['A'], (1, 2), None),
     ('label', 'A', {'supp.': ('A', 1), 'unsupp.': (1, 2)}, 'unsupp.'),
     ('label', ['A'], {'supp.': ('A', 1), 'unsupp.': (1, 2)}, 'unsupp.')],
    ids=['no value_condition, string data', 'no value_condition, list data',
         'value_condition, string data', 'value_condition, list data'])
def test_validate_value_in_set_negative(label, data, options, value_condition):
    """Test the 'validate_value_in_set' function with unsupported input."""

    # Assert the raise of an exception
    with raises(ValueError):
        validate_value_in_set(label, data, options, value_condition)
