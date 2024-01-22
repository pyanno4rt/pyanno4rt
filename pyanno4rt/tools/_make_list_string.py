"""Text-to-string-list conversion."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def make_list_string(text, min_length):
    """."""

    # Replace round brackets with square brackets
    text.replace('(', '[')
    text.replace(')', ']')

    # Check if the text has a minimum length and is not bracketed
    if len(text) > min_length and (text[0], text[-1]) != ('[', ']'):

        # Return the text input with square brackets
        return ''.join(('[', text, ']'))

    else:

        # Return the (already bracketed) text input
        return text
