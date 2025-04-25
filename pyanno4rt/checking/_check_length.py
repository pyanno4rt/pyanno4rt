"""Vector length checking."""

# Author: Tim Ortkamp

# %% External package import

from numpy import ndarray
from operator import eq, ge, gt, le, lt

# %% Function definition


def check_length(label, data, reference, sign):
    """
    Check if the length of a value is correct.

    Parameters
    ----------
    label : str
        Label for the item to be checked.

    data : list, tuple or ndarray
        Input value to be checked.

    reference : int
        Reference value for the length comparison.

    sign : {'==', '>', '>=', '<', '<='}
        Sign for the length comparison.

    Raises
    ------
    ValueError
        If the value has an incorrect length.
    """

    # Map the operators
    operator_dict = {'==': eq, '>=': ge, '>': gt, '<=': le, '<': lt}

    # Check if the value is iterable but the length is incorrect
    if (isinstance(data, (list, tuple, ndarray))
            and not operator_dict[sign](len(data), reference)):

        # Raise an error
        raise ValueError(
            f"The treatment plan parameter '{label}' has length "
            f"{len(data)}, but should be {sign} {reference}!")
