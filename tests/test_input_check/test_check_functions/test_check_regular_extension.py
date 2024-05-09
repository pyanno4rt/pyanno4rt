"""File regularity and extension check function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.input_check.check_functions import check_regular_extension

# %% Test definition


# Define the valid argument sets
@mark.parametrize(
    'label, data, extensions',
    [('label', './tests/extra_files/load_list_from_file/list.json',
      ('.json', '.p', '.txt')),
     ('label', './tests/extra_files/load_list_from_file/list.p',
      ('.json', '.p', '.txt')),
     ('label', './tests/extra_files/load_list_from_file/list.txt',
      ('.json', '.p', '.txt'))],
    ids=['JSON', 'Python binary', 'text file'])
def test_check_regular_extension_valid(label, data, extensions):
    """Test the 'check_regular_extension' function with valid input."""

    # Assert the run-through of the function
    assert check_regular_extension(label, data, extensions) is None


# Define the invalid argument sets
@mark.parametrize(
    'label, data, extensions, expected',
    [('label', './tests/extra_files/load_list_from_file.txt', ('.p', '.txt'),
      FileNotFoundError),
     ('label', './tests/extra_files/load_list_from_file/list.json', ('.txt',),
      TypeError),
     ('label', './tests/extra_files/load_list_from_file/list.json',
      ('.p', '.txt'), TypeError)],
    ids=['directory', 'single-set', 'multi-set'])
def test_check_regular_extension_invalid(label, data, extensions, expected):
    """Test the 'check_regular_extension' function with invalid input."""

    # Assert the raise of an error exception
    with raises(expected):
        check_regular_extension(label, data, extensions)
