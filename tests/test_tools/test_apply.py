"""Apply function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from pytest import mark

# %% Internal package import

from pyanno4rt.tools import apply

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'function, elements, returns, expected',
    [(lambda x: x**2, [1, 2, 3], True, [1, 4, 9]),
     (lambda x: x**2, [1, 2, 3], False, None)],
    ids=['square function with return', 'square function without return'])
def test_apply(function, elements, returns, expected):
    """Test the 'apply' function."""

    # Assert the equality between actual and expected outcome
    assert apply(function, elements, returns) == expected
