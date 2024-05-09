"""Directory regularity and extension check function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.input_check.check_functions import (
    check_regular_extension_directory)

# %% Test definition


def test_check_regular_extension_directory_valid():
    """
    Test the 'check_regular_extension_directory' function with valid input.
    """

    # Assert the run-through of the function
    assert check_regular_extension_directory(
        'label', './tests/extra_files/load_list_from_file',
        ('.json', '.p', '.py', '.txt')) is None


# Define the invalid argument sets
@mark.parametrize(
    'label, data, extensions, expected',
    [('label', './tests/extra_files/load_list_from_file', ('.txt',),
      TypeError),
     ('label', './tests/extra_files/load_list_from_file', ('.p', '.txt'),
      TypeError),
     ('label', './tests/extra_files/load_list_from_file/list.json', ('.txt',),
      NotADirectoryError)],
    ids=['single-set', 'multi-set', 'invalid directory'])
def test_check_regular_extension_directory_invalid(label, data, extensions,
                                                   expected):
    """
    Test the 'check_regular_extension_directory' function with invalid input.
    """

    # Assert the raise of an error exception
    with raises(expected):
        check_regular_extension_directory(label, data, extensions)
