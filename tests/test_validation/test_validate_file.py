"""File validation function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.validation import validate_file

# %% Test definition


# Define the supported argument sets
@mark.parametrize(
    'label, data, options',
    [('label', './tests/extra_files/load_list_from_file/list.json',
      ('.json', '.p', '.txt')),
     ('label', './tests/extra_files/load_list_from_file/list.p',
      ('.json', '.p', '.txt')),
     ('label', './tests/extra_files/load_list_from_file/list.txt',
      ('.json', '.p', '.txt'))],
    ids=['JSON', 'Python binary', 'text file'])
def test_validate_file_positive(label, data, options):
    """Test the 'validate_file' function with supported input."""

    # Assert the run-through of the function
    assert validate_file(label, data, options) is None


# Define the unsupported argument sets
@mark.parametrize(
    'label, data, options, expected',
    [('label', './tests/extra_files/load_list_from_file.txt', ('.p', '.txt'),
      FileNotFoundError),
     ('label', './tests/extra_files/load_list_from_file/list.json', ('.txt',),
      TypeError),
     ('label', './tests/extra_files/load_list_from_file/list.json',
      ('.p', '.txt'), TypeError)],
    ids=['directory', 'single-set', 'multi-set'])
def test_validate_file_negative(label, data, options, expected):
    """Test the 'validate_file' function with unsupported input."""

    # Assert the raise of an exception
    with raises(expected):
        validate_file(label, data, options)
