"""Directory validation function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.validation import validate_directory

# %% Test definition


def test_validate_directory_positive():
    """
    Test the 'validate_directory' function with valid input.
    """

    # Assert the run-through of the function
    assert validate_directory(
        'label', './tests/extra_files/load_list_from_file',
        ('.json', '.p', '.py', '.txt'), ('.mat', '.p')) is None


# Define the invalid argument sets
@mark.parametrize(
    'label, item, options, expected',
    [
     ('label', './tests/extra_files/load_list_from_file', ('.txt',),
      TypeError),
     ('label', './tests/extra_files/load_list_from_file', ('.p', '.txt'),
      TypeError),
     ('label', './tests/extra_files/load_list_from_file/list.json', ('.txt',),
      NotADirectoryError)
     ],
    ids=[
        'single-set',
        'multi-set',
        'not a directory'
        ]
    )
def test_validate_directory_negative(label, item, options, expected):
    """
    Test the 'validate_directory' function with invalid input.
    """

    # Assert the raise of an exception
    with raises(expected):
        validate_directory(label, item, options, ('.mat', '.p'))
