"""Deduplication function test."""

# Author: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pytest import mark

# %% Internal package import

from pyanno4rt.tools import deduplicate

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'elements, expected',
    [((1, 2, 3, 4, 1), {1: [0, 4], 2: [1], 3: [2], 4: [3]}),
     (('hello', 'hello', 1.5, True), {'hello': [0, 1], 1.5: [2], True: [3]})],
    ids=['integer keys', 'multi-type keys'])
def test_deduplicate(elements, expected):
    """Test the 'deduplicate' function."""

    # Assert the equality between actual and expected outcome
    assert deduplicate(elements) == expected
