"""Dictionary comparison function test."""

# Author: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import array
from pytest import mark

# %% Internal package import

from pyanno4rt.tools import compare_dictionaries

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'reference_dict, compare_dict, expected',
    [({'A': 1}, {'B': 1}, False),
     ({'A': 1}, {'A': 1.0}, False),
     ({'A': array([1])}, {'A': array([0])}, False),
     ({'A': 1}, {'A': 2}, False),
     ({'A': 1}, {'A': 1}, True)],
    ids=['different keys', 'different value type', 'different value array',
         'different value', 'equal dictionaries'])
def test_compare_dictionaries(reference_dict, compare_dict, expected):
    """Test the 'compare_dictionaries' function."""

    # Assert the equality between actual and expected outcomes
    assert compare_dictionaries(reference_dict, compare_dict) == expected
