"""Text square bracketing."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def add_square_brackets(text):
    """
    Add square brackets to a string-type text.

    Parameters
    ----------
    text : str
        Input text to be placed in brackets.

    Returns
    -------
    text : str
        Input text with enclosing square brackets (if non-empty string).
    """

    # Check if the input text is non-empty
    if len(text) > 0:

        # Check if the first character is a bracket
        if text[0] in ('(', '[', '{'):

            # Remove the first character
            text = text[1:]

        # Check if the last character is a bracket
        if text[-1] in (')', ']', '}'):

            # Remove the last character
            text = text[:-1]

        # Return the text input only
        return f'[{text}]'

    return text
