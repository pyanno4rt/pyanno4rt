"""Dictionary comparison function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from numpy import array
from pytest import mark

# %% Internal package import

from pyanno4rt.tools import compare_dictionaries

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'reference_dict, compare_dict, ignore_keys, expected',
    [({'A': 1, 'B': 2}, {'A': 1, 'B': 2, 'C': 3}, ['C'], True),
     ({'A': 1}, {'B': 1}, None, False),
     ({'A': 1}, {'A': 1.0}, None, False),
     ({'A': array([1])}, {'A': array([0])}, None, False),
     ({'A': 1}, {'A': 2}, None, False),
     ({'A': 1}, {'A': 1}, None, True)],
    ids=['ignore keys', 'different keys', 'different value type',
         'different value array', 'different value', 'equal dictionaries'])
def test_compare_dictionaries(
        reference_dict, compare_dict, ignore_keys, expected):
    """Test the 'compare_dictionaries' function."""

    # Assert the equality between actual and expected outcomes
    assert (
        compare_dictionaries(reference_dict, compare_dict, ignore_keys)
        == expected)
