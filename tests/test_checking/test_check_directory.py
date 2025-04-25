"""Directory check function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.checking import check_directory

# %% Test definition


def test_check_directory_positive():
    """
    Test the 'check_directory' function with supported input.
    """

    # Assert the run-through of the function
    assert check_directory(
        'label', './tests/extra_files/load_list_from_file',
        ('.json', '.p', '.py', '.txt'), ('.mat', '.p')) is None


# Define the unsupported argument sets
@mark.parametrize(
    'label, data, options, expected',
    [('label', './tests/extra_files/load_list_from_file', ('.txt',),
      TypeError),
     ('label', './tests/extra_files/load_list_from_file', ('.p', '.txt'),
      TypeError),
     ('label', './tests/extra_files/load_list_from_file/list.json', ('.txt',),
      NotADirectoryError)],
    ids=['single-set', 'multi-set', 'not a directory'])
def test_check_directory_negative(label, data, options, expected):
    """
    Test the 'check_directory' function with unsupported input.
    """

    # Assert the raise of an exception
    with raises(expected):
        check_directory(label, data, options, ('.mat', '.p'))
