"""Extension checking in folder."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from os import listdir
from os.path import isdir

# %% Function definition


def check_extensions_in_folder(key, value, extensions):
    """Check if the path extensions of the folder files are not valid."""

    # Check if any file extension is not valid w.r.t the extensions
    if (isdir(value) and not all(
            [any([file.endswith(extension) for extension in extensions])
             for file in listdir(value)])):

        # Check if multiple extensions should be checked
        if len(extensions) == 1:

            # Get the extension string from the element
            extensions_string = extensions[0]

        else:

            # Get the error string for the extensions
            extensions_string = " or ".join(
                (", ".join(extensions[:-1]), extensions[-1]))

        # Raise an error to indicate a wrong path extension
        raise TypeError(
            f"The treatment plan parameter '{key}' leads to a valid "
            f"directory, but not all files end with {extensions_string}!")
