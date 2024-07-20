"""NaN replacement function test."""

# Author: Moritz Müller, Tim Ortkamp <tim.ortkamp.kit.edu>

# %% External package import

from numpy import nan
from pytest import mark

# %% Internal package import

from pyanno4rt.tools import replace_nan

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'elements, value, expected',
    [((1, 2, nan, 4, True), 3, (1, 2, 3, 4, True)),
     ((nan,), [1, 2, 3, 4], ([1, 2, 3, 4],))],
    ids=['integer', 'list'])
def test_replace_nan(elements, value, expected):
    """Test the 'replace_nan' function."""

    # Assert the equality between actual and expected outcome
    assert tuple(replace_nan(elements, value)) == expected
