"""Arange with endpoint function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import array, array_equal
from pytest import mark

# %% Internal package import

from pyanno4rt.tools import arange_with_endpoint

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'start, stop, step, expected',
    [(0, 2, 1, array([0, 1, 2])), (0, 2.5, 1, array([0.0, 1.0, 2.0]))],
    ids=['integer', 'decimal'])
def test_arange_with_endpoint(start, stop, step, expected):
    """Test the 'arange_with_endpoint' function."""

    # Assert the equality between actual and expected outcome
    assert array_equal(arange_with_endpoint(start, stop, step), expected)
