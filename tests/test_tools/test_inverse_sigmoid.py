"""Inverse sigmoid function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pytest import mark

# %% Internal package import

from pyanno4rt.tools import inverse_sigmoid

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'value, multiplier, summand, expected',
    [(0.5, 1, 0, 0.0), ((0.5,), 10, 10, (-1.0,))],
    ids=['single-value, standard', 'multi-value, parameterized'])
def test_inverse_sigmoid(value, multiplier, summand, expected):
    """Test the 'inverse_sigmoid' function."""

    # Assert the equality between actual and expected outcome
    assert inverse_sigmoid(value, multiplier, summand) == expected
