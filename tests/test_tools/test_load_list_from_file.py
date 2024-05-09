"""Load list from file function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pytest import mark

# %% Internal package import

from pyanno4rt.tools import load_list_from_file

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'path, expected',
    [('./tests/extra_files/load_list_from_file/list.json', list(range(1, 11))),
     ('./tests/extra_files/load_list_from_file/list.p', list(range(1, 11))),
     ('./tests/extra_files/load_list_from_file/list.txt', list(range(1, 11)))],
    ids=['JSON', 'Python binary', 'text file'])
def test_load_list_from_file(path, expected):
    """Test the 'load_list_from_file' function."""

    # Assert the equality between actual and expected outcome
    assert load_list_from_file(path) == expected
