"""Non-increasing function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import array
from pytest import mark

# %% Internal package import

from pyanno4rt.tools import non_increasing

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'input_array, expected',
    [(array([1, 2, 3]), False), (array([1, 3, 2]), False),
     (array([3, 2, 1]), True), (array([2, 1, 1]), True)],
    ids=['increasing', 'semi-increasing', 'decreasing', 'decreasing/constant'])
def test_non_increasing(input_array, expected):
    """Test the 'non_increasing' function."""

    # Assert the equality between actual and expected outcome
    assert non_increasing(input_array) == expected
