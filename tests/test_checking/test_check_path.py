"""Path check function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.checking import check_path

# %% Test definition


# Define the supported argument sets
@mark.parametrize(
    'label, data',
    [('label', './tests/extra_files/load_list_from_file/list.json'),
     ('label', './tests/extra_files/load_list_from_file')],
    ids=['file path', 'directory path'])
def test_check_path_positive(label, data):
    """Test the 'check_path' function with supported input."""

    # Assert the run-through of the function
    assert check_path(label, data) is None


# Define the unsupported argument sets
@mark.parametrize(
    'label, data',
    [('label', 'not/a/real/file/path.py'),
     ('label', 'not/a/real/directory/folder')],
    ids=['file path', 'directory path'])
def test_check_path_negative(label, data):
    """Test the 'check_path' function with unsupported input."""

    # Assert the raise of an exception
    with raises(IOError):
        check_path(label, data)
