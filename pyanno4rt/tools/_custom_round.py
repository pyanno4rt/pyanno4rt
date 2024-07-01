"""Custom number rounding."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def custom_round(number):
    """
    Round up a number from 5 as the first decimal place, otherwise round down.

    Parameters
    ----------
    number : int or float
        The number to be rounded.

    Returns
    -------
    float
        The rounded number.
    """

    # Convert the number into string and trim after the first decimal place
    number = str(float(number))[:str(float(number)).index('.') + 2]

    # Check if the last digit is equal or greater than '5'
    if number[-1] >= '5':

        # Return the rounded up number
        return float(number[:-3] + str(int(number[-3]) + 1))

    # Otherwise, just cut off the last digit
    return float(number[:-1])
