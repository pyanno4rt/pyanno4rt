"""Objectives return."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Function definition


def get_all_objectives(segmentation):
    """
    Return a tuple with the user-assigned objectives.

    Parameters
    ----------
    segmentation : dict
        Dictionary with information on the segmented structures.

    Returns
    -------
    tuple
        Tuple with the user-assigned objectives.
    """

    return tuple(objective for objective in flatten(
        segmentation[segment]['objective'] for segment in segmentation
        if segmentation[segment]['objective'] is not None))
