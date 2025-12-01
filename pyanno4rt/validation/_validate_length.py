"""Iterable length validation."""

# Author: Tim Ortkamp

# %% External package import

from operator import eq, ge, gt, le, lt

from numpy import ndarray

# %% Function definition


def validate_length(label, item, reference, sign):
    """
    Validation function for the item length.

    Parameters
    ----------
    label : str
        Label for the validation item.

    item : list, tuple or ndarray
        Validation item.

    reference : int
        Reference value.

    sign : {'==', '>', '>=', '<', '<='}
        Symbol for the comparison sign.

    Raises
    ------
    ValueError
        If the item has an invalid length.
    """

    # Map the operators
    operator_dict = {'==': eq, '>=': ge, '>': gt, '<=': le, '<': lt}

    # Check if an iterable with invalid length has been passed
    if (isinstance(item, (list, tuple, ndarray))
            and not operator_dict[sign](len(item), reference)):

        # Raise an error
        raise ValueError(
            f"The treatment plan parameter '{label}' has length "
            f"{len(item)}, but should be {sign} {reference}!")
