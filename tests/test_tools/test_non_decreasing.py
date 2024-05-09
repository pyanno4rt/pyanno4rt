"""Non-decreasing function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import array
from pytest import mark

# %% Internal package import

from pyanno4rt.tools import non_decreasing

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'input_array, expected',
    [(array([1, 2, 3]), True), (array([1, 3, 2]), False),
     (array([3, 2, 1]), False), (array([1, 2, 2]), True)],
    ids=['increasing', 'semi-decreasing', 'decreasing', 'increasing/constant'])
def test_non_decreasing(input_array, expected):
    """Test the 'non_decreasing' function."""

    # Assert the equality between actual and expected outcome
    assert non_decreasing(input_array) == expected
