"""Identity function test."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.tools import identity

# %% Test definition


def test_identity():
    """Test the identity function."""

    # Set the expected outcome
    expected = [0, 0.0, '0', True, [0], (0,), {'0': 0}]

    # Get the actual outcome
    actual = [identity(x, None) for x in expected]

    # Assert the equality between actual and expected outcome
    assert actual == expected
