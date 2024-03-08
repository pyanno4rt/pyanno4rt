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
    str
        Input text with enclosing square brackets.
    """

    # Check if the input text is non-empty
    if len(text) > 0:

        # Check if the first character is a round bracket
        if text[0] == '(':

            # Replace it with a square bracket
            text = text.replace('(', '[', 1)

        # Check if the last character is a round bracket
        if text[-1] == ')':

            # Replace it with a square bracket
            text = text.replace(')', ']', 1)

        # Check if the text is not already bracketed
        if (text[0], text[-1]) != ('[', ']'):

            # Return the text input with square brackets
            return ''.join(('[', text, ']'))

        else:

            # Return the text input only
            return text

    return text
