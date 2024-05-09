"""Monotonic function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import array
from pytest import mark

# %% Internal package import

from pyanno4rt.tools import monotonic

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'input_array, expected',
    [(array([1, 2, 3]), True), (array([1, 3, 2]), False),
     (array([3, 2, 1]), True), (array([1, 2, 2]), True)],
    ids=['monotonically increasing', 'non-monotonic',
         'monotonically decreasing', 'monotonically increasing/constant'])
def test_monotonic(input_array, expected):
    """Test the 'monotonic' function."""

    # Assert the equality between actual and expected outcome
    assert monotonic(input_array) == expected
