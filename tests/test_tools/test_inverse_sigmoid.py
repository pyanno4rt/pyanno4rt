"""Inverse sigmoid function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from math import inf
from pytest import mark

# %% Internal package import

from pyanno4rt.tools import inverse_sigmoid

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'value, multiplier, summand, expected',
    [(0.5, 1, 0, 0.0), ((0.5,), 10, 10, (-1.0,)), (0, 1, 0, -inf),
     (1, 1, 0, inf)],
    ids=['single-value, standard', 'multi-value, parameterized',
         'single-value, lower bound', 'single-value, upper bound'])
def test_inverse_sigmoid(value, multiplier, summand, expected):
    """Test the 'inverse_sigmoid' function."""

    # Assert the equality between actual and expected outcome
    assert inverse_sigmoid(value, multiplier, summand) == expected
