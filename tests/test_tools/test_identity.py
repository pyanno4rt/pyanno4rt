"""Identity function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from pytest import mark

# %% Internal package import

from pyanno4rt.tools import identity

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'value, expected',
    [(0, 0), ('0', '0'), (True, True), ({'0': 0}, {'0': 0})],
    ids=['integer', 'string', 'boolean', 'dictionary'])
def test_identity(value, expected):
    """Test the 'identity' function."""

    # Assert the equality between actual and expected outcome
    assert identity(value) == expected
