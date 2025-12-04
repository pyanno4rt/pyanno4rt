"""Path validation function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from pytest import mark, raises

# %% Internal package import

from pyanno4rt.validation import validate_path

# %% Test definition


# Define the valid argument sets
@mark.parametrize(
    'label, item',
    [
     ('label', './tests/extra_files/load_list_from_file/list.json'),
     ('label', './tests/extra_files/load_list_from_file')
     ],
    ids=[
        'file path',
        'directory path'
        ]
    )
def test_validate_path_positive(label, item):
    """Test the 'validate_path' function with valid input."""

    # Assert the run-through of the function
    assert validate_path(label, item) is None


# Define the invalid argument sets
@mark.parametrize(
    'label, item',
    [
     ('label', 'not/a/real/file/path.py'),
     ('label', 'not/a/real/directory/folder')
     ],
    ids=[
        'file path',
        'directory path'
        ]
    )
def test_validate_path_negative(label, item):
    """Test the 'validate_path' function with invalid input."""

    # Assert the raise of an exception
    with raises(IOError):
        validate_path(label, item)
