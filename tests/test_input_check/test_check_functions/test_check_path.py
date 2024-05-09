"""Path check function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.input_check.check_functions import check_path

# %% Test definition


# Define the valid argument sets
@mark.parametrize(
    'label, data',
    [('label', './tests/extra_files/load_list_from_file/list.json'),
     ('label', './tests/extra_files/load_list_from_file')],
    ids=['file path', 'directory path'])
def test_check_path_valid(label, data):
    """Test the 'check_path' function with valid input."""

    # Assert the run-through of the function
    assert check_path(label, data) is None


# Define the invalid argument sets
@mark.parametrize(
    'label, data',
    [('label', 'not/a/real/file/path.py'),
     ('label', 'not/a/real/directory/folder')],
    ids=['file path', 'directory path'])
def test_check_path_invalid(label, data):
    """Test the 'check_path' function with invalid input."""

    # Assert the raise of an IOError exception
    with raises(IOError):
        check_path(label, data)
