"""Custom rounding function test."""

# Author: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pytest import mark

# %% Internal package import

from pyanno4rt.tools import custom_round

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'number, expected',
    [(3.543, 4.0), (3, 3.0), (3.323, 3.0)],
    ids=['round up', 'no round', 'round down'])
def test_custom_round(number, expected):
    """Test the 'custom_round' function."""

    # Assert the equality between actual and expected outcome
    assert custom_round(number) == expected
