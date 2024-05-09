"""Sigmoid function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pytest import mark

# %% Internal package import

from pyanno4rt.tools import sigmoid

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'value, multiplier, summand, expected',
    [(0, 0, 0, 0.5), ((0, 0), 10, 0, (0.5, 0.5))],
    ids=['single', 'multi'])
def test_sigmoid(value, multiplier, summand, expected):
    """Test the 'sigmoid' function."""

    # Assert the equality between actual and expected outcome
    assert sigmoid(value, multiplier, summand) == expected
