"""Text-to-string-list conversion."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def make_list_string(text, min_length):
    """."""

    # 
    text.replace('(', '[')
    text.replace(')', ']')

    # 
    if len(text) > min_length and (text[0], text[-1]) != ('[', ']'):

        # 
        return ''.join(('[', text, ']'))

    else:

        # 
        return text
