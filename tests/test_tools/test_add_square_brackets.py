"""Text square bracketing function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pytest import mark

# %% Internal package import

from pyanno4rt.tools import add_square_brackets

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'text, expected',
    [('', ''), (' ', '[ ]'), ('A', '[A]'), ('(A)', '[A]'), ('[A]', '[A]')],
    ids=['empty', 'spacing', 'single', 'rounded', 'squared'])
def test_add_square_brackets(text, expected):
    """Test the 'add_square_brackets' function."""

    # Assert the equality between actual and expected outcome
    assert add_square_brackets(text) == expected
