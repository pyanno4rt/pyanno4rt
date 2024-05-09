"""Flattening function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pytest import mark

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'elements, expected',
    [([1, [2], [3, [4]], "String", [5], [[6]]], (1, 2, 3, 4, 'String', 5, 6)),
     ([], ()), ([b'Hello', [b'World', [b'!']]], (b'Hello', b'World', b'!'))],
    ids=['nested', 'empty', 'byte'])
def test_flatten(elements, expected):
    """Test the 'flatten' function."""

    # Assert the equality between actual and expected outcome
    assert tuple(flatten(elements)) == expected
