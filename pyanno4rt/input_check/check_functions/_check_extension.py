"""Extension checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from os.path import isfile

# %% Function definition


def check_extension(key, value, extensions):
    """Check if the file path extension is not valid."""

    # Check if the file extension is not valid w.r.t the extensions
    if (isfile(value) and not any(value.endswith(extension)
                                  for extension in extensions)):

        # Raise an error to indicate a wrong path extension
        raise TypeError(
            "The treatment plan parameter '{}' leads to a valid file, but it "
            "does not end with {}!"
            .format(key,
                    extensions[0]
                    if len(extensions) == 1
                    else
                    " or ".join((", ".join(extensions[:-1]), extensions[-1]))))
