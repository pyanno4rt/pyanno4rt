"""Objectives return."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Function definition


def get_objectives(segmentation):
    """
    Return a tuple with the user-assigned objectives.

    Parameters
    ----------
    segmentation : dict
        Dictionary with information on the segmented structures.

    Returns
    -------
    tuple
        Flattened tuple with the user-assigned objectives.
    """

    return tuple(flatten(segmentation[segment]['objective']
                         for segment in segmentation
                         if segmentation[segment]['objective']))
